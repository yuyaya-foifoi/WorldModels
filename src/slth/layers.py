import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

DenseConv = nn.Conv2d
DenseDeconv = nn.ConvTranspose2d


def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    """
    @staticmethod
    def backward(ctx, g):
        return g, None
    """

    @staticmethod
    def backward(ctx, g):
        return g, None


class SubsetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias = None
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class SubnetTransposeConv(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet

        x = F.conv_transpose2d(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return x


class GetQuantnet_binary(autograd.Function):
    @staticmethod
    def forward(ctx, scores, weights, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory. switched 0 and 1
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # Perform binary quantization of weights
        abs_wgt = torch.abs(
            weights.clone()
        )  # Absolute value of original weights
        q_weight = abs_wgt * out  # Remove pruned weights
        num_unpruned = int(k * scores.numel())  # Number of unpruned weights
        alpha = (
            torch.sum(q_weight) / num_unpruned
        )  # Compute alpha = || q_weight ||_1 / (number of unpruned weights)

        # Save absolute value of weights for backward
        ctx.save_for_backward(abs_wgt)

        # Return pruning mask with gain term alpha for binary weights
        return alpha * out

    @staticmethod
    def backward(ctx, g):
        # Get absolute value of weights from saved ctx
        (abs_wgt,) = ctx.saved_tensors
        # send the gradient g times abs_wgt on the backward pass
        return g * abs_wgt, None, None


class QuantnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias = None
        self.abs_score = True
        self.subnet_func = GetQuantnet_binary

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_kaiming_normal":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def forward(self, x):
        quantnet = GetQuantnet_binary.apply(
            self.clamped_scores, self.weight, self.remain_rate
        )
        w = torch.sign(self.weight) * quantnet
        return F.linear(x, w, self.bias)


# Not learning weights, finding subnet
class QuantnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetQuantnet_binary

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_kaiming_normal":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)

    def forward(self, x):
        quantnet = GetQuantnet_binary.apply(
            self.clamped_scores, self.weight, self.remain_rate
        )
        # Binarize weights by taking sign, multiply by pruning mask and gain term (alpha)
        w = torch.sign(self.weight) * quantnet
        # Pass binary subnetwork weights to convolution layer
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


class QuantnetTransposeConv(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetQuantnet_binary

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_kaiming_normal":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)

    def forward(self, x):
        quantnet = GetQuantnet_binary.apply(
            self.clamped_scores, self.weight, self.remain_rate
        )
        # Binarize weights by taking sign, multiply by pruning mask and gain term (alpha)
        w = torch.sign(self.weight) * quantnet
        # Pass binary subnetwork weights to convolution layer
        x = F.conv_transpose2d(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return x


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


class SingleShotLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        params = (self.in_features, self.out_features, self.weight)
        self.scores = nn.Parameter(linear_layer_synflow_scoring(params))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias = None
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


# Not learning weights, finding subnet
class SingleShotConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x,
            w,
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
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.abs_score = True
        self.subnet_func = GetSubnet

    @property
    def clamped_scores(self):
        if self.abs_score:
            return self.scores.abs()
        else:
            self.scores.data = F.relu(self.scores.data)
            return self.scores

    def set_remain_rate(self, remain_rate):
        self.remain_rate = remain_rate

    def init_weight(self, name=None):
        if name is None:
            name = "signed_constant"
        self._init_weight(self.weight, name=name)

    def _init_weight(self, weight, name="signed_constant"):
        if name == "signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

        elif name == "scaled_signed_constant":
            fan = nn.init._calculate_correct_fan(weight, mode="fan_in")
            fan = fan * (1 - self.remain_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            weight.data = weight.data.sign() * std

    def forward(self, x):
        subnet = self.subnet_func.apply(self.clamped_scores, self.remain_rate)
        w = self.weight * subnet

        x = F.conv_transpose2d(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return x
