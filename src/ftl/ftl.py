import copy
import os

import torch


def get_FTL_weight(
    model,
    log_dir: str,
    pth_name: str,
    FTL_key_list: list = [],
    FTL_ratio: float = None,
    avoid_FTL_key_list: list = [],
):

    init_weight = copy.deepcopy(model.state_dict())
    target_weight = copy.deepcopy(model.state_dict())
    source_weight = torch.load(os.path.join(log_dir, pth_name))
    state_dict_keys = list(model.state_dict().keys())

    for key in state_dict_keys:
        target_weight[key] += source_weight[key]

        if key.split(".")[0] in FTL_key_list:
            target_weight[key] += source_weight[key] * FTL_ratio

        if key.split(".")[0] in avoid_FTL_key_list:
            target_weight[key] = target_weight[key]

    assert (
        torch.sum(
            init_weight[state_dict_keys[0]]
            == target_weight[state_dict_keys[0]]
        ).item()
        == 0
    ), "Some of the weights do not seem to have changed after the transfer. You may want to check."

    return target_weight


def get_FTL_score(
    model,
    log_dir: str,
    pth_name: str,
    FTL_key_list: list = [],
    FTL_ratio: float = None,
    avoid_FTL_key_list: list = [],
):

    init_weight = copy.deepcopy(model.state_dict())
    target_weight = copy.deepcopy(model.state_dict())
    source_weight = torch.load(os.path.join(log_dir, pth_name))
    state_dict_keys = [
        k
        for k in list(model.state_dict().keys())
        if k.split(".")[-1] == "scores"
    ]

    for key in state_dict_keys:
        source_weight[key] += target_weight[key]

        if key.split(".")[0] in FTL_key_list:
            source_weight[key] = (
                target_weight[key] + source_weight[key] * FTL_ratio
            )

        if key.split(".")[0] in avoid_FTL_key_list:
            source_weight[key] = target_weight[key]

    assert (
        torch.sum(
            init_weight[state_dict_keys[0]]
            == source_weight[state_dict_keys[0]]
        ).item()
        == 0
    ), "Some of the weights do not seem to have changed after the transfer. You may want to check."

    return target_weight
