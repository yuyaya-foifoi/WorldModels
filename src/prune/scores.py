import torch
import torch.nn as nn


def get_score(model) -> dict:
    scores = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            scores[name] = torch.abs(layer.weight.grad) * layer.weight
    return scores


def get_ftl_score(
    source_scores: dict,
    target_scores: dict,
    full_transfer_keys: list,
    fractional_transfer_keys: list,
    omega: float = 0.2,
) -> dict:

    omega = omega

    ftl_scores = {}
    for k, v in target_scores.items():
        if k in full_transfer_keys:
            ftl_scores[k] = target_scores[k] + source_scores[k]
        elif k in fractional_transfer_keys:
            ftl_scores[k] = target_scores[k] + omega * source_scores[k]
        else:
            pass
    return ftl_scores
