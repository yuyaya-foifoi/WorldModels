#from src.model.model_base import (VGG,resnet,image_resnet,lenet)
from src.model.vgg import VGG
from src.model.resnet import resnet
from src.model.image_resnet import image_resnet
from src.model.lenet import lenet


def get_network(network, depth, dataset, use_bn=True):
    if network == 'vgg':
        print('Use batch norm is: %s' % use_bn)
        return VGG(depth=depth, dataset=dataset, batchnorm=use_bn)
    elif network == 'resnet':
        if (depth - 2) % 6 == 0:
            return resnet(depth=depth, dataset=dataset)
        else:
            return image_resnet(depth=depth, dataset=dataset)
    elif network == 'lenet':
        return lenet(dataset=dataset)
    else:
        raise NotImplementedError('Network unsupported ' + network)


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)
