# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
pruning.
@author: AnonymousUser314156
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import copy
import types
from tqdm import tqdm
#from pruner.panning import RLPanning, Panning

from src.utils.prune_utils import reset_mask, fetch_data, linearize, nonlinearize, get_keep_ratio
from src.utils.matplot_utils import PltClass


def get_connected_scores(keep_masks, info='', mode=0, verbose=0):
    _connected_scores = 0
    _last_filter = None
    for m, g in keep_masks.items():
        if isinstance(m, nn.Conv2d) and 'padding' in str(m):  # Does not consider the shortcut part of resnet
            # [n, c, k, k]
            _2d = np.sum(np.abs(keep_masks[m].cpu().detach().numpy()), axis=(2, 3))
            _channel = np.sum(_2d, axis=0)  # vector

            if _last_filter is not None:  # The first layer is not considered
                for i in range(_channel.shape[0]):  # Traverse channel filter
                    if _last_filter[i] == 0 and _channel[i] != 0:
                        _connected_scores += 1
                    if mode == 1:
                        if _last_filter[i] != 0 and _channel[i] == 0:
                            _connected_scores += 1

            _last_filter = np.sum(_2d, axis=1)

    if verbose == 1:
        print(f'{info}-{mode}->connected scores: {_connected_scores}')
    return _connected_scores


def coincide_mask(mask1, mask2):
    _coin_num = 0
    for m, s in mask1.items():
        _coin_num += torch.sum(mask1[m]*mask2[m] == 1)
    _m2_num = torch.sum(torch.cat([torch.flatten(x == 1) for x in mask2.values()]))
    return (_coin_num/_m2_num).cpu().detach().numpy()


def masks_compare(mb, masks, trainloader, device, inf_str=''):
    pr_str = '-'*20
    pr_str += '\n%s' % inf_str if len(inf_str) > 0 else ''
    mb.register_mask(masks)
    mb_ratios = mb.get_ratio_at_each_layer()
    pr_str += '\n** %s - Remaining: %.5f%%' % ('', mb_ratios['ratio'])
    true_masks = effective_masks_synflow(mb.model, masks, trainloader, device)
    mb.register_mask(true_masks)
    effective_ratios = mb.get_ratio_at_each_layer()
    pr_str += '\n** %s - Remaining: %.5f%%\n' % ('true_masks', effective_ratios['ratio'])
    pr_str += '-'*20
    print(pr_str)
    return pr_str, mb_ratios['ratio'], effective_ratios['ratio']


def print_mask_information(mb, logger=None, inf_str=''):
    ratios = mb.get_ratio_at_each_layer()
    if logger:
        logger.info('** %s - Mask information of %s. Overall Remaining: %.5f%%' % (inf_str, mb.get_name(), ratios['ratio']))
    re_str = '** %s - Mask information of %s. Overall Remaining: %.5f%%\n' % (inf_str, mb.get_name(), ratios['ratio'])
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        if logger:
            logger.info('  (%d) %s: Remaining: %.5f%%' % (count, k, v))
        re_str += '  (%d) %.5f%%\n' % (count, v)
        count += 1
    return re_str


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
def hessian_gradient_product(net, samples, device, config, T=200, reinit=False):
    """
        data_mode:
            0 - Grouping with different labels
            1 - Grouping with same labels
        gard_mode:
            0 - Gradient norm gradient (GraSP)
            3 - Sensitive to loss (SNIP)
    """

    data_mode = config.data_mode
    grad_mode = config.grad_mode
    num_group = config.num_group
    samples_per_class = config.samples_per_class
    num_classes = config.classe

    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    # print("fetch data")
    if samples_per_class == 5 and num_group == 2:
        samples_per_class = 6
    inputs, targets = samples if isinstance(samples, tuple) else fetch_data(samples, num_classes, samples_per_class, data_mode)
    equal_parts = inputs.shape[0] // num_group
    inputs = inputs.to(device)
    targets = targets.to(device)
    gradg_list = []

    if grad_mode == 0:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            _grad = autograd.grad(_loss, weights, create_graph=True)
            _gz = 0
            _layer = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    _gz += _grad[_layer].pow(2).sum()  # g * g
                    _layer += 1
            gradg_list.append(autograd.grad(_gz, weights))
    elif grad_mode == 3:
        for i in range(num_group):
            _outputs = net.forward(inputs[i * equal_parts:(i + 1) * equal_parts]) / T
            _loss = F.cross_entropy(_outputs, targets[i * equal_parts:(i + 1) * equal_parts])
            gradg_list.append(autograd.grad(_loss, weights, retain_graph=True))

    return gradg_list


def ranking_mask(scores, keep_ratio, normalize=True, eff_rank=False, verbose=0, acc_score=None, oir_mask=None):
    eps = 1e-10
    keep_masks = dict()
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    if oir_mask is not None:
        min_score = torch.min(all_scores)
        # Minimize deleted scores to avoid re-selecting
        for m, g in scores.items():
            scores[m][oir_mask[m] == 0] = min_score
            all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    norm_factor = norm_factor if normalize else torch.ones_like(norm_factor)
    all_scores.div_(norm_factor)
    num_params_to_rm = int(len(all_scores) * keep_ratio)
    threshold, _index = torch.topk(all_scores, num_params_to_rm)
    acceptable_score = threshold[-1] if acc_score is None else acc_score
    for m, g in scores.items():
        if eff_rank:
            keep_masks[m] = ((g / norm_factor) != 0).float()
        else:
            keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()
    if verbose == 1:
        print("** norm factor:", norm_factor)
        print('** accept: ', acceptable_score)
    elif verbose == 2:
        return keep_masks, acceptable_score
    elif verbose == 3:
        return keep_masks, acceptable_score, scores
    return keep_masks


def Single_ranking_pruning(net, ratio, samples, device, config=None, reinit=False, retrun_inf=0, verbose=1, oir_mask=None, oir_w=None):
    """
    :param retrun_inf: Other return parameters, 0: return connectivity, 1: return sorting score, 2: score and threshold
    """
    if ratio == 0:
        return reset_mask(net), 0

    keep_ratio = (1 - ratio)
    old_net = net
    net = copy.deepcopy(net)  # .eval()
    net.train() if config.train_mode == 1 else net.eval()
    net.zero_grad()

    num_iters = config.get('num_iters', 1)
    score_mode = config.score_mode

    if score_mode == 0:
        return reset_mask(net), 0

    gradg_list = None
    for it in range(num_iters):
        if verbose == 1:
            print("Iterations %d/%d." % (it, num_iters))
        sample_n = (samples[0][it], samples[1][it]) if isinstance(samples, tuple) else samples
        _hessian_grad = hessian_gradient_product(net, sample_n, device, config, reinit=reinit)
        if gradg_list is None:
            gradg_list = _hessian_grad
        else:
            for i in range(len(gradg_list)):
                _grad_i = _hessian_grad[i]
                gradg_list[i] = [gradg_list[i][_l] + _grad_i[_l] for _l in range(len(_grad_i))]
    # print(len(gradg_list))

    # === 剪枝部分 ===
    """
        score_mode:
            1 - and
            2 - and absolute value
            3 - product
            4 - Euclidean distance

        grasp: --grad_mode 0 --score_mode 1
        snip: --grad_mode 3 --score_mode 2
    """

    # === Calculate the score ===
    layer_cnt = 0
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            kxt = 0
            weight = oir_w[old_modules[idx]] if oir_w is not None else layer.weight.data
            if score_mode == 1:
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
            elif score_mode == 2:
                for i in range(len(gradg_list)):
                    _qhg = weight * gradg_list[i][layer_cnt]  # theta_q grad
                    kxt += _qhg
                kxt = torch.abs(kxt)
            grads[old_modules[idx]] = kxt
            layer_cnt += 1

    if retrun_inf == 9:
        return grads

    # === get masks ===
    keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)
    if verbose == 1:
        print('** accept: ', threshold)
        print('Remaining:', torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    _connected_scores = get_connected_scores(keep_masks, f"{'-' * 20}\nBefore", 1, verbose)

    if retrun_inf == 0:
        return keep_masks, _connected_scores
    elif retrun_inf == 2:
        return keep_masks, grads, threshold
    else:
        return keep_masks, grads


def Iterative_pruning(model, ratio, trainloader, device, config):

    old_net = model
    old_modules = list(old_net.modules())
    keep_masks = reset_mask(old_net)
    net = copy.deepcopy(model)
    net.train()
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    num_iters = config.num_iters_prune
    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        # keep_ratio = 1.0 - ratio * ((epoch + 1) / num_iters)
        keep_ratio = (1.0 - ratio) ** ((epoch + 1) / num_iters)

        net.zero_grad()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = copy_net_weights[old_modules[idx]] * keep_masks[old_modules[idx]]

        score_mode = config.score_mode
        gradg_list = hessian_gradient_product(net, trainloader, device, config)

        layer_cnt = 0
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                kxt = 0
                _w = copy_net_weights[old_modules[idx]] if config.dynamic else layer.weight.data
                if score_mode == 1:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                elif score_mode == 2:
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt += _qhg
                    kxt = torch.abs(kxt)
                elif score_mode == 3:
                    kxt = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt]  # theta_q grad
                        kxt *= torch.abs(_qhg)
                elif score_mode == 4:
                    aef = 1e6
                    for i in range(len(gradg_list)):
                        _qhg = _w * gradg_list[i][layer_cnt] * aef  # theta_q grad
                        kxt += _qhg.pow(2)
                    kxt = kxt.sqrt()
                grads[old_modules[idx]] = kxt
                layer_cnt += 1

        oir_mask = None if config.dynamic else keep_masks
        keep_masks, threshold = ranking_mask(grads, keep_ratio, verbose=2, oir_mask=oir_mask)

        if 'fig' in config.debug:
            if epoch == 0:
                fig = PltClass(500, 500, 60)

            if epoch+1 >= 10:
                fig.plt_end()
            else:
                if epoch%2 == 0:
                    fig.plt_score(grads, 9, int(epoch/2), f'{(1 - keep_ratio) * 100:0.2f}%')
                    fig.plt_inset_axes(grads, 9, int(epoch/2))


        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, threshold))
            prog_bar.set_description(desc, refresh=True)

    return keep_masks, 0


# Based on https://github.com/ganguli-lab/Synaptic-Flow/blob/master/Pruners/pruners.py#L178
def SynFlow(model, ratio, dataloader, device, num_iters, eff_rank=False, ori_masks=None, dynamic=False):

    old_net = model
    net = copy.deepcopy(model)
    net.eval()
    net.zero_grad()
    modules_ls = list(old_net.modules())
    signs = linearize(net)
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(device)
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)
    keep_masks = reset_mask(old_net) if ori_masks is None else ori_masks

    desc = ('[keep ratio=%s] acceptable score: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio)**((epoch + 1) / num_iters)
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = (copy_net_weights[modules_ls[idx]] * keep_masks[modules_ls[idx]]).abs_()
        net.zero_grad()
        # forward
        output = net(input)
        torch.sum(output).backward()
        # synflow score
        grads = dict()
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = copy_net_weights[modules_ls[idx]] if dynamic else layer.weight.data
                grads[modules_ls[idx]] = (_w * layer.weight.grad).abs_()  # theta g
        # synflow masks
        keep_masks, acceptable_score = ranking_mask(grads, keep_ratio, False, eff_rank, 2)

        if num_iters > 1:
            desc = ('[keep ratio=%s] acceptable score: %e' % (keep_ratio, acceptable_score))
            prog_bar.set_description(desc, refresh=True)

    # nonlinearize(net, signs)

    return keep_masks, 0


# Based on https://github.com/avysogorets/Effective-Sparsity/blob/main/effective_masks.py#28
def effective_masks_synflow(model, masks, trainloader, device):
    """ computes effective sparsity of a pruned model using SynFlow
    """
    true_masks, _ = SynFlow(model, 0, trainloader, device, 1, True, masks)
    return true_masks


def Pruner(mb, trainloader, device, config):
    net = mb.model
    """
        config.prune_mode: 
            dense, rank, rank/random, rank/iterative, panning
    """
    if 'dense' in config.prune_mode:
        masks, _ = Single_ranking_pruning(net, 0, None, None, config)
    elif 'rank' in config.prune_mode:
        if config.rank_algo.lower() == 'synflow':
            masks, _ = SynFlow(net, config.target_ratio, trainloader, device, config.num_iters_prune, dynamic=config.dynamic)
        else:
            if 'iterative' in config.prune_mode:
                masks, _ = Iterative_pruning(net, config.target_ratio, trainloader, device, config)
            else:
                masks, _score = Single_ranking_pruning(net, config.target_ratio, trainloader, device, config, reinit=True, retrun_inf=1)
        if 'random' in config.prune_mode:
            for m, g in masks.items():
                perm = torch.randperm(g.nelement())
                masks[m] = g.reshape(-1)[perm].reshape(g.shape)
            get_connected_scores(masks, 'rank/random', 1)
    # elif 'panning' in config.prune_mode:
    #     if 'rl' in config.prune_mode:
    #         masks, _ = RLPanning(net, trainloader, device, config)
    #     else:
    #         masks, _ = Panning(net, trainloader, device, config)
    else:
        raise NotImplementedError('Prune mode unsupported ' + config.prune_mode)

    return masks, 0
