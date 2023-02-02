# -*- coding: utf-8 -*-

"""
Created on 05/10/2022
prune_utils.
@author: AnonymousUser314156
"""

import torch
import torch.nn as nn


def get_keep_ratio(masks, layer_ratio=False):
    _l_r = []
    _remain_num = 0
    _all_num = 0
    for m, s in masks.items():
        _re_num = torch.sum(masks[m] == 1)
        _a_num = masks[m].numel()
        _remain_num += _re_num
        _all_num += _a_num
        if layer_ratio:
            _l_r.append((_re_num / _a_num).cpu().detach().numpy())
    if layer_ratio:
        return (_remain_num / _all_num).cpu().detach().numpy(), _l_r
    else:
        return (_remain_num / _all_num).cpu().detach().numpy()


@torch.no_grad()
def linearize(model):
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs


@torch.no_grad()
def nonlinearize(model, signs):
    for name, param in model.state_dict().items():
        param.mul_(signs[name])


def reset_mask(net):
    keep_masks = dict()
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            keep_masks[layer] = torch.ones_like(layer.weight.data).float()
    return keep_masks


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L11
def fetch_data(dataloader, num_classes, samples_per_class, dm=0):
    if dm == 9:
        # dataloader_iter = iter(dataloader)
        # inputs, targets = next(dataloader_iter)
        for inputs, targets in dataloader:
            X, y = inputs[0:samples_per_class * num_classes], targets[0:samples_per_class * num_classes]
            break
    else:
        datas = [[] for _ in range(num_classes)]
        labels = [[] for _ in range(num_classes)]
        mark = dict()

        # dataloader_iter = iter(dataloader)
        # while True:
        #     inputs, targets = next(dataloader_iter)

        for inputs, targets in dataloader:
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
                category = y.item()
                if len(datas[category]) == samples_per_class:
                    mark[category] = True
                    continue
                datas[category].append(x)
                labels[category].append(y)
            if len(mark) == num_classes:
                break

        X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)

        if dm == 1:  # different label groups
            _index = []
            for i in range(samples_per_class):
                _index.extend([i + j * samples_per_class for j in range(0, num_classes)])
            X = X[_index]
            y = y[_index]

    return X, y



class FetchData:
    """
    fetch data
    """
    def __init__(self, dataloader, device, num_classes, samples_per_class, dm=0):
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.dm = dm

    def samples(self):
        x, y = fetch_data(self.dataloader, self.num_classes, self.samples_per_class, self.dm)
        return x.to(self.device), y.to(self.device)
