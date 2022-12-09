# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def init_distributed(port=40101, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('distributed training not available')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception:
        world_size, rank = 1, 0
        logger.info('distributed training not available')

    return world_size, rank


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
            return new_lr / self.ref_lr

        # -- progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.T_max))
        new_lr = max(self.final_lr,
                     self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))
        return new_lr / self.ref_lr


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads

# make a random sampled buffer
def make_buffer_lst(buffer_lst, buffer_size, subset_path, subset_path_cls, tasks, task_idx):
    import random
    # Get the index in dataset for labeled samples of current task
    def get_lst(subset_path, subset_path_cls, target_cls):
        buffer_lst = []
        cls_idx_lst = []
        cls_lst = []
        with open(subset_path_cls, 'r') as rfile:
            for i, line in enumerate(rfile):
                label = int(line.split('\n')[0])
                if label in target_cls:
                    cls_idx_lst.append(i)
                    cls_lst.append(label)
        index_lst = []
        with open(subset_path, 'r') as rfile:
            for i, line in enumerate(rfile):
                index = int(line.split('\n')[0])
                if i in cls_idx_lst:
                    index_lst.append(index)
        return cls_lst, cls_idx_lst, index_lst

    pre_classes = sum(tasks[:task_idx], [])
    num_pre_classes = len(pre_classes)
    seen_classes = sum(tasks[:task_idx+1], [])
    num_seen_classes = len(seen_classes)
    num_cur_classes = len(tasks[task_idx])
    cls_lst, cls_idx_lst, index_lst = get_lst(subset_path, subset_path_cls, tasks[task_idx])
    sorted_idx = np.argsort(cls_lst)
    cls_lst = np.array(cls_lst)[sorted_idx].tolist()
    index_lst = np.array(index_lst)[sorted_idx].tolist()
    if buffer_lst is None:
        if buffer_size >= len(index_lst):
            buffer_lst = index_lst
        else:
            cur_num_per_class = int(len(index_lst)/num_cur_classes)
            new_num_per_class = int(buffer_size / num_seen_classes)
            assert new_num_per_class <= cur_num_per_class
            buffer_lst = []
            for i in range(num_seen_classes):
                buffer_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][:new_num_per_class]
            # Fill the empty space of buffer
            if len(buffer_lst) < buffer_size:
                diff = buffer_size - len(buffer_lst)
                for i in range(min(num_cur_classes, diff)):
                    buffer_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][-1:]
    else:
        num_in_buffer = len(buffer_lst)
        if buffer_size - num_in_buffer >= len(index_lst):
            buffer_lst += index_lst
        else:
            cur_num_per_class = int(len(index_lst)/num_cur_classes)
            pre_num_per_class = int(len(buffer_lst)/num_pre_classes)
            new_num_per_class = int(buffer_size / num_seen_classes)
            num_modulo = buffer_size - pre_num_per_class * num_pre_classes
            assert new_num_per_class <= cur_num_per_class
            assert new_num_per_class <= pre_num_per_class
            temp_lst = []
            for i in range(num_pre_classes):
                temp_lst += buffer_lst[i*pre_num_per_class:(i+1)*pre_num_per_class][:new_num_per_class]
            for i in range(num_cur_classes):
                temp_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][:new_num_per_class]      
            # Fill the empty space of buffer
            if len(temp_lst) < buffer_size:
                diff = buffer_size - len(temp_lst)
                for i in range(min(num_pre_classes, diff)):
                    temp_lst += buffer_lst[i*pre_num_per_class:(i+1)*pre_num_per_class][-1:]
            if len(temp_lst) < buffer_size:
                diff = buffer_size - len(temp_lst)
                for i in range(min(num_cur_classes, diff)):
                    temp_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][-1:]    
            buffer_lst = temp_lst

    return buffer_lst
