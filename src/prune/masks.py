import torch
import torch.nn as nn


def get_masks(scores: dict, keep_ratio: float) -> dict:
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = {}
    for k, v in scores.items():
        masks = ((v / norm_factor) >= acceptable_score).float()
        assert v.shape == masks.shape
        keep_masks[k] = masks
    return keep_masks


def hook_factory(keep_mask: torch.tensor):
    """
    The hook function can't be defined directly here because of Python's
    late binding which would result in all hooks getting the very last
    mask! Getting it through another function forces early binding.
    """

    def hook(grads):
        return grads * keep_mask

    return hook


def apply_masks(model, keep_masks: dict):

    keep_masks_keys = list(keep_masks.keys())

    for k, layer in model.named_modules():
        if k in keep_masks_keys:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                keep_mask = keep_masks[k]
                """
                len([n for n in encoder.modules()])
                > 
                5

                """
                assert layer.weight.shape == keep_mask.shape
                layer.weight.data[keep_mask == 0.0] = 0.0
                layer.weight.register_hook(hook_factory(keep_mask))
