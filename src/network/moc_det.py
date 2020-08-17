from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}


class MOC_Backbone(nn.Module):
    def __init__(self, arch, num_layers,):
        super(MOC_Backbone, self).__init__()
        self.backbone = backbone[arch](num_layers)

    def forward(self, input):
        return self.backbone(input)


class MOC_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K, flip_test=False):
        super(MOC_Det, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.branch = MOC_Branch(backbone.backbone.output_channel, arch, head_conv, branch_info, K)

    def forward(self, chunk1, chunk2):
        assert(self.K == len(chunk1))
        if self.flip_test:
            assert(self.K == len(chunk2))
            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            return [self.branch(chunk1)]
