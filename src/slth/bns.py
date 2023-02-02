import torch.nn as nn


class NonAffineBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, dim):
        super(NonAffineBatchNorm1d, self).__init__(dim, affine=False)


class NonAffineBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm2d, self).__init__(dim, affine=False)
