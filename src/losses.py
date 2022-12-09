# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
from src.utils import (
    AllGather,
    AllReduce
)

logger = getLogger()


def init_paws_loss(
    multicrop=6,
    tau=0.1,
    T=0.25,
    me_max=True,
    return_probs=False
):
    """
    Make semi-supervised PAWS loss

    :param multicrop: number of small multi-crop views
    :param tau: cosine similarity temperature
    :param T: target sharpenning temperature
    :param me_max: whether to perform me-max regularization
    """
    softmax = torch.nn.Softmax(dim=1)

    def sharpen(p):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: normalize embeddings
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

        # Step 2: gather embeddings from all workers
        supports = AllGather.apply(supports)

        # Step 3: compute similarlity between local embeddings
        return softmax(query @ supports.T / tau) @ labels

    def loss(
        anchor_views,
        anchor_supports,
        anchor_support_labels,
        target_views,
        target_supports,
        target_support_labels,
        sharpen=sharpen,
        snn=snn,
        mask=None
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        batch_size = len(anchor_views) // (2+multicrop)

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = snn(target_views, target_supports, target_support_labels)
            targets = sharpen(targets)
            if multicrop > 0:
                mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
                targets = torch.cat([targets, *[mc_target for _ in range(multicrop)]], dim=0)
            targets[targets < 1e-4] *= 0  # numerical stability

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = AllReduce.apply(torch.mean(sharpen(probs), dim=0))
            rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))

        return loss, rloss, probs

    return loss, snn, sharpen


def make_labels_matrix(
    num_classes,
    s_batch_size,
    world_size,
    device,
    unique_classes=False,
    smoothing=0.0,
    task_idx=None
):
    """
    Make one-hot labels matrix for labeled samples

    NOTE: Assumes labeled data is loaded with ClassStratifiedSampler from
          src/data_manager.py
    """

    local_images = s_batch_size*num_classes
    total_images = local_images*world_size

    off_value = smoothing/(num_classes*world_size) if unique_classes else smoothing/num_classes

    if unique_classes:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for r in range(world_size):
            # -- index range for rank 'r' images
            s1 = r * local_images
            e1 = s1 + local_images
            # -- index offset for rank 'r' classes
            offset = r * num_classes
            for i in range(num_classes):
                labels[s1:e1][i::num_classes][:, offset+i] = 1. - smoothing + off_value
    else:
        labels = torch.zeros(total_images, num_classes*world_size).to(device) + off_value
        for i in range(num_classes):
            labels[i::num_classes][:, i] = 1. - smoothing + off_value

    return labels


def gather_from_all(tensor):
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def gather_tensors_from_all(tensor):
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    ):
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def convert_to_distributed_tensor(tensor):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = 'cpu' if not tensor.is_cuda else 'gpu'
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor, orig_device):
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == 'cpu':
        tensor = tensor.cpu()
    return tensor
