import torch
import torch.nn as nn

from . import bns, layers


def recursive_setattr(obj, name, value):
    if "." in name:
        parent, child = name.split(".")[0], ".".join(name.split(".")[1:])
        recursive_setattr(getattr(obj, parent), child, value)
    else:
        setattr(obj, name, value)


def modify_module_for_slth(
    model,
    remain_rate,
    omit_prefix_list: list = [],
    init_mode: str = None,
    abs_score: bool = True,
    is_subnet_conv: bool = True,
):
    named_modules = [(n, l) for n, l in model.named_modules()]
    print("#Modules: {}".format(len(named_modules)))
    for name, layer in named_modules:
        if name in omit_prefix_list:
            layer.weight.requires_grad = False
        elif isinstance(layer, nn.Conv2d):

            if is_subnet_conv:
                print("Replace nn.Conv2d with SubnetConv: {}".format(name))
                slth_layer = layers.SubnetConv(
                    layer.in_channels,
                    layer.out_channels,
                    stride=layer.stride,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    bias=True,
                )
            else:
                print("Replace nn.Conv2d with QuantnetConv: {}".format(name))
                slth_layer = layers.QuantnetConv(
                    layer.in_channels,
                    layer.out_channels,
                    stride=layer.stride,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    bias=True,
                )
            slth_layer.set_remain_rate(remain_rate)
            slth_layer.init_weight(init_mode)
            recursive_setattr(model, name, slth_layer)
        elif isinstance(layer, nn.Linear):
            if is_subnet_conv:
                print("Replace nn.Linear with SubnetLinear: {}".format(name))
                slth_layer = layers.SubsetLinear(
                    layer.in_features, layer.out_features, bias=False
                )
            else:
                print("Replace nn.Linear with QuantnetLinear: {}".format(name))
                slth_layer = layers.QuantnetLinear(
                    layer.in_features, layer.out_features, bias=False
                )

            slth_layer.set_remain_rate(remain_rate)
            slth_layer.init_weight(init_mode)
            recursive_setattr(model, name, slth_layer)
        elif isinstance(layer, nn.BatchNorm1d):
            print(
                "Replace nn.BatchNorm1d with NonAffineBatchNorm1d: {}".format(
                    name
                )
            )
            slth_layer = bns.NonAffineBatchNorm1d(dim=layer.num_features)
            recursive_setattr(model, name, slth_layer)
        elif isinstance(layer, nn.BatchNorm2d):
            print(
                "Replace nn.BatchNorm2d with NonAffineBatchNorm2d: {}".format(
                    name
                )
            )
            slth_layer = bns.NonAffineBatchNorm2d(dim=layer.num_features)
            recursive_setattr(model, name, slth_layer)
        else:
            print("No modification", name, type(layer))
    return model
