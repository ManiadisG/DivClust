import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from utils.misc import export_fn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=None):
        super(BasicBlock, self).__init__()
        assert (track_running_stats is not None)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet34Model(nn.Module):
    def __init__(self, clusters=10, clusterings=1):
        super(ResNet34Model, self).__init__()
        self.sobel = self._make_sobel_()
        self.inplanes = 64
        self.clusters = clusters
        self.clusterings = clusterings
        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(64, track_running_stats=True), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.layer2 = self._make_layer(BasicBlock, 64, 3)
        self.layer3 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer5 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.linear_heads = nn.ModuleList([nn.Linear(512, clusters) for p in range(clusterings)])
        self._initialise_weights_()

    def run(self, x, target=None):
        if target is None or target > 5:
            raise NotImplementedError('Target is expected to be smaller than 6')
        layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
        for layer in layers[:target]:
            x = layer(x)
        return x

    def forward(self, x, softmax=False, return_features=False):
        if self.sobel is not None:
            x = self.sobel(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        features = x
        x = torch.stack([self.linear_heads[k](x) for k in range(self.clusterings)])
        if softmax:
            x = F.softmax(x, -1)
        if return_features:
            return x, features
        else:
            return x

    def _initialise_weights_(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is None:
                    continue
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, track_running_stats=True))

        return nn.Sequential(*layers)

    def _make_sobel_(self):
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
        sobel_filter.weight.data[1, 0].copy_(torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
        sobel_filter.bias.data.zero_()
        layers = nn.Sequential(grayscale, sobel_filter)
        for p in layers.parameters():
            p.requires_grad = False
        return layers

    def get_backbone_parameters(self):
        params = []
        for m in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            params += list(m.parameters())
        return params

    def get_projection_head_parameters(self):
        return list(self.linear_heads.parameters())

@export_fn
def PICA_ResNet34(clusters=10, clusterings=1, *args, **kwargs):
    return ResNet34Model(clusters, clusterings)