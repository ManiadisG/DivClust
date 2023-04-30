import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import export_fn
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, widen=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, layer0_Conv3=False, layer0_add_pooling=True, final_pooling="avg", *args, **kwargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        num_out_filters = width_per_group * widen
        if layer0_Conv3:
            layer0_kernel, layer0_stride, layer0_padding = 3, 1, 1
        else:
            layer0_kernel, layer0_stride, layer0_padding = 7, 2, 3
        self.layer0 = self._make_layer_0(num_out_filters, norm_layer, layer0_kernel, layer0_stride, layer0_padding,
                                         layer0_add_pooling)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(block, num_out_filters, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        self.layer3 = self._make_layer(block, num_out_filters, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        self.layer4 = self._make_layer(block, num_out_filters, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if final_pooling == "avg":
            self.final_pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif final_pooling == "max":
            self.final_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.output_shape = self.inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer_0(self, num_out_filters, norm_layer, kernel_size=7, stride=2, padding=3, pooling=True):
        n_layer = norm_layer(num_out_filters)
        if pooling:
            p_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            p_layer = nn.Identity()
        return nn.Sequential(nn.Conv2d(3, num_out_filters, kernel_size=kernel_size, stride=stride, padding=padding,
                                       bias=False), n_layer, nn.ReLU(inplace=True), p_layer)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, return_fmaps=None):
        if return_fmaps is not None and return_fmaps == "last":
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x
        elif return_fmaps is not None and return_fmaps == "all":
            fmaps = [self.layer0(x)]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                fmaps.append(layer(fmaps[-1]))
            return fmaps[1:]
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.final_pooling(x)
            x = torch.flatten(x, 1)
            return x


@export_fn
def resnet18(*args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2],*args, **kwargs)
    
@export_fn
def resnet34(*args, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], *args, **kwargs)


@export_fn
def resnet50(*args, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], *args, **kwargs)
