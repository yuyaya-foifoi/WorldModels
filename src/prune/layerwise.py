import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_layer_synflow_scoring(params):
    in_features, out_features, weight = params
    input = torch.ones(in_features, out_features).T
    initial_weight = torch.clone(weight.detach())
    weight = nn.Parameter(weight.abs())
    loss = torch.sum(F.linear(input, weight))
    loss.backward()
    scores = torch.mul(weight.grad.data, initial_weight)
    return scores


def linear_conv_synflow_scoring(params, img_size=64):
    in_channels, weight, bias, stride, padding, dilation, groups = params
    input = torch.ones(in_channels, img_size, img_size)
    initial_weight = torch.clone(weight.detach())
    weight = nn.Parameter(weight.abs())
    loss = torch.sum(
        F.conv2d(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
    )
    loss.backward()
    scores = torch.mul(weight.grad.data, initial_weight)
    return scores


def linear_transpose_conv_synflow_scoring(params, img_size=64):
    (
        in_channels,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    ) = params
    input = torch.ones(in_channels, img_size, img_size)
    initial_weight = torch.clone(weight.detach())
    weight = nn.Parameter(weight.abs())
    loss = torch.sum(
        F.conv_transpose2d(
            input=input,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
    )
    loss.backward()
    scores = torch.mul(weight.grad.data, initial_weight)
    return scores


def get_masks(scores: torch.tensor, keep_ratio: float) -> dict:
    all_scores = torch.cat([torch.flatten(x) for x in scores])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    masks = ((scores / norm_factor) >= acceptable_score).float()
    assert scores.shape == masks.shape
    return masks


def hook_factory(keep_mask: torch.tensor):
    """
    The hook function can't be defined directly here because of Python's
    late binding which would result in all hooks getting the very last
    mask! Getting it through another function forces early binding.
    """

    def hook(grads):
        return grads * keep_mask

    return hook


def apply_mask(weight: torch.tensor, mask: torch.tensor):

    assert weight.shape == mask.shape
    weight.data[mask == 0.0] = 0.0
    weight.register_hook(hook_factory(mask))


class SingleShotLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.keep_ratio = 0.3
        params = (self.in_features, self.out_features, self.weight)
        self.scores = nn.Parameter(linear_layer_synflow_scoring(params))
        self.apply_mask_to_weight()
        self.weight.requires_grad = False
        self.bias = None

    def apply_mask_to_weight(self):
        mask = get_masks(self.scores, self.keep_ratio)
        apply_mask(self.weight, mask)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class SingleShotConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.keep_ratio = 0.3
        params = (
            self.in_channels,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.scores = nn.Parameter(linear_conv_synflow_scoring(params))
        self.apply_mask_to_weight()
        self.weight.requires_grad = False
        self.bias = None

    def apply_mask_to_weight(self):
        mask = get_masks(self.scores, self.keep_ratio)
        apply_mask(self.weight, mask)

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class SingleShotTransposeConv(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.keep_ratio = 0.3
        params = (
            self.in_channels,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        self.scores = nn.Parameter(
            linear_transpose_conv_synflow_scoring(params)
        )
        self.apply_mask_to_weight()
        self.weight.requires_grad = False
        self.bias = None

    def apply_mask_to_weight(self):
        mask = get_masks(self.scores, self.keep_ratio)
        apply_mask(self.weight, mask)

    def forward(self, x):
        x = F.conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return x


def recursive_setattr(obj, name, value):
    if "." in name:
        parent, child = name.split(".")[0], ".".join(name.split(".")[1:])
        recursive_setattr(getattr(obj, parent), child, value)
    else:
        setattr(obj, name, value)


def modify_module_for_layer_wise_single_shot_pruning(
    model,
):
    named_modules = [(n, l) for n, l in model.named_modules()]
    print("#Modules: {}".format(len(named_modules)))
    for name, layer in named_modules:
        if isinstance(layer, nn.Conv2d):
            print("Replace nn.Conv2d with SingleShotConv: {}".format(name))
            new_layer = SingleShotConv(
                layer.in_channels,
                layer.out_channels,
                stride=layer.stride,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                bias=True,
            )

            recursive_setattr(model, name, new_layer)

        elif isinstance(layer, nn.Linear):
            print("Replace nn.Linear with SingleShotLinear: {}".format(name))
            new_layer = SingleShotLinear(
                layer.in_features, layer.out_features, bias=False
            )
            recursive_setattr(model, name, new_layer)

        elif isinstance(layer, nn.ConvTranspose2d):

            print(
                "Replace nn.ConvTranspose2d with SingleShotTransposeConv: {}".format(
                    name
                )
            )
            new_layer = SingleShotTransposeConv(
                layer.in_channels,
                layer.out_channels,
                stride=layer.stride,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                bias=True,
            )
            recursive_setattr(model, name, new_layer)

    return model
