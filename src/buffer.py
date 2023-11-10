# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, min_per_cls=1):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'features', 'task_labels']
        self.all_cls = []
        self.min_per_cls = min_per_cls
        self.num_each_cls = {}
        

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, features: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param features: tensor containing the features of the inputs
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device) - 1)

    def add_data(self, examples, labels=None, logits=None, features=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param features: tensor containing the features of the inputs
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, features, task_labels)
        # Special case for buffer size == 0
        # NOTE: If want to try very small buffer size, e.g., 8 or 16, need to force 
        # the buffer to keep a balanced number of samples for each class
        if self.buffer_size == 0:
            self.examples = examples.to(self.device)
            if labels is not None:
                self.labels = labels.to(self.device)
            if logits is not None:
                self.logits = logits.to(self.device)
            if features is not None:
                self.features = features.to(self.device)
            if task_labels is not None:
                self.task_labels= task_labels.to(self.device)
            return
        # Check if there is new class
        add_classes = torch.unique(labels)
        new_class = [c for c in add_classes if c not in self.all_cls]
        self.all_cls += new_class

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)

            # Force the data for new classes to be added
            if len(new_class) > 0 and labels[i] in new_class:
                if index < 0:
                    index = np.random.randint(0, self.buffer_size)
                    while self.labels[index].item() != labels[i].item() and self.num_each_cls[self.labels[index].item()] <= self.min_per_cls:
                        index = np.random.randint(0, self.buffer_size)

            # Force a min number of samples for each class (for very small buffer size)
            if self.labels[index].item() != -1 and self.num_each_cls[self.labels[index].item()] == self.min_per_cls:
                continue

            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    # -1 for removed samples
                    if self.labels[index] != -1:
                        self.num_each_cls[self.labels[index].item()] -= 1

                    self.labels[index] = labels[i].to(self.device)
                    # +1 for added samples
                    if labels[i].item() in self.num_each_cls:
                        self.num_each_cls[labels[i].item()] += 1
                    else:
                        self.num_each_cls[labels[i].item()] = 1
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if features is not None:
                    self.features[index] = features[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, mask_task=-1, cpt=None, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        if mask_task > 0:
            masked_examples = self.examples[self.labels // cpt != mask_task]
        else:
            masked_examples = self.examples

        choice = np.random.choice(min(self.num_seen_examples, masked_examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in masked_examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def get_equal_data(self, size: int, mask_task=-1, cpt=None, transform: transforms=None, sup_views=1):
        # NOTE: If want to try very small buffer size, e.g., 8 or 16, need to force
        # the buffer to keep a balanced number of samples for each class
        if self.buffer_size == 0:
            return self.get_batch_data(transform, sup_views)
        
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            select_size = min(self.num_seen_examples, self.examples.shape[0])
        else:
            select_size = size
        if mask_task > 0:
            masked_examples = self.examples[self.labels // cpt != mask_task]
        else:
            masked_examples = self.examples

        if self.labels is None:
            choice = np.random.choice(min(self.num_seen_examples, masked_examples.shape[0]),
                                  size=select_size, replace=True)
        else:
            classes = torch.unique(self.labels)

            classes = classes[classes>=0] # Removing the place-holder class
            assert len(classes) * self.min_per_cls <= self.buffer_size
            number_per_class = size // len(classes)
            can = []
            for c in classes:
                ### Trying a faster version: Much better!
                selected_idx = self.labels == c
                if selected_idx.sum() > 1:
                    idx_lst = selected_idx.nonzero().squeeze()
                else:
                    idx_lst = selected_idx.nonzero()
                idx_temp = idx_lst 
                while len(idx_lst) < number_per_class:
                    idx_lst  = torch.cat((idx_lst, idx_temp))
                idx = torch.randperm(idx_lst.size(0))[:number_per_class]                
                can.append(idx_lst[idx])

            # Shuffle the  classes TODO: is this necessary?
            idx_lst = np.arange(len(classes))
            np.random.shuffle(idx_lst)
            can = [can[idx_lst[i]] for i in range(len(classes))]
            # make the class distribution as (a, b, c, a, b, c)
            choice = []
            for idx in zip(*can):
                choice += idx
            choice = [c.item() for c in choice]
        if transform is None: transform = lambda x: transforms.ToTensor()(x)
        ex_list = []
        for _ in range(sup_views):
            ex_list.append(torch.stack([transform(transforms.ToPILImage()(ee.cpu()))
                            for ee in masked_examples[choice]]).to(self.device))

        ret_dict = {}
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                attr_list = torch.cat([attr[choice] for _ in range(sup_views)]).to(self.device)
                ret_dict[attr_str] = attr_list
        return ex_list, ret_dict
    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device)[:self.num_seen_examples],)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)[:self.num_seen_examples]
                ret_tuple += (attr,)
        return ret_tuple

    def get_batch_data(self, transform: transforms=None, sup_views=1) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        
        ex_list = []
        for _ in range(sup_views):
            ex_list.append(torch.stack([transform(transforms.ToPILImage()(ee.cpu()))
                            for ee in self.examples]).to(self.device))
        ret_dict = {}
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                attr_list = torch.cat([attr for _ in range(sup_views)]).to(self.device)
                ret_dict[attr_str] = attr_list 

        return ex_list, ret_dict
    
    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
