import torch
import torchvision.models
from torch import nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor
from sketch_core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point
from typing import Type, Any, Callable, Union, List, Optional
import torch_scatter
import sketch_core.models.resnet2d as resnet2d
import sketch_core.models.mobilenet as mobilenet
import sketch_core.models.efficientnet as efficientnet
import sketch_core.models.mnasnet as mnasnet
import sketch_core.models.swin_transformerv2 as swin
import sketch_core.models.mbv2 as mbv2
import sketch_core.models.inceptionv3 as inv3
import sketch_core.models.densenet as densenet
import sketch_core.models.mobilenetnv3 as mobilenetnv3
from einops import rearrange, repeat
import torch.nn.functional as F
import copy
from .gnn import GNN_module

__all__ = ['SD3B', 'SD3B_ResNet18Merge', "SD3B_ResNet34Merge", "SD3B_ResNet50Merge", "SD3B_ResNet101Merge",
           "SD3B_Mbv3Merge", "SD3B_EffMerge", "SD3B_NasMerge",  "SD3B_SwinMerge",
           "SD3B_Mbv2Merge", "SD3B_Incv3Merge", "SD3B_DenseMerge", "SD3B_ResNet152Merge", "SD3B_SwinSolo",
           "SD3B_ResNet18Solo", "SD3B_ResNet34Solo", "SD3B_ResNet50Solo", "SD3B_ResNet101Solo", "SD3B_ResNet152Solo",
           "SD3B_Mbv3HMerge", "Trans_ResNet50Merge", "RNN_ResNet50Merge", "RNN_ResNet50Norm", "SD3B_ResNet50Norm", "Trans_ResNet50Norm",
           "GNN_ResNet50Norm", "SD3B2_ResNet50Norm", "SD3B1_ResNet50Norm"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return spnn.Conv3d(in_planes, out_planes, kernel_size=2 if stride==2 else 3, stride=stride, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return spnn.Conv3d(in_planes, out_planes, kernel_size=2 if stride==2 else 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = spnn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = spnn.ReLU(True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.eca = ECAAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = spnn.BatchNorm
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = spnn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = spnn.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 64, layers[0])
        if len(layers) >= 2:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
        if len(layers) >= 3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        if len(layers) >= 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # def _forward_impl(self, x):
    #     # See note [TorchScript super()]
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     return x
    #
    # def forward(self, x):
    #     return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2], pretrained, progress,
                   **kwargs)

def resnet4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2], pretrained, progress,
                   **kwargs)

def resnet6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2], pretrained, progress,
                   **kwargs)
class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class SD3B(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(6, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))

        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        out = self.classifier(pooled_data)
        return out


class SD3B_ResNet18Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet18Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet18(pretrained=True)
        # self.img_branch_postprocess = resnet2d.conv1x1(2048, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))

        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out

class SD3B_ResNet18Solo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet18Solo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.img_branch = resnet2d.resnet18(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out

class SD3B_ResNet34Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet34Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet34(pretrained=True)
        # self.img_branch_postprocess = resnet2d.conv1x1(2048, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))

        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        ### first interaction
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)

        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = img_out + pooled_data
        out = self.classifier(merge_out)
        return out


class SD3B_ResNet34Solo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet34Solo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.img_branch = resnet2d.resnet34(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        ### first interaction
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out

class SD3B_ResNet50Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet50Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(7, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        # print("*****************", x.F.shape)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data

class SD3B1_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B1_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet2()
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier_voxel = nn.Sequential(nn.Linear(64, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        # print("*****************", x.F.shape)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        z1 = voxel_to_point(x1, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out


class SD3B2_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B2_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet4()
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier_voxel = nn.Sequential(nn.Linear(128, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        # print("*****************", x.F.shape)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        # x3 = self.net.layer3(x2)
        # x4 = self.net.layer4(x3)
        # # x4 = self.process(x4)
        z1 = voxel_to_point(x2, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out


class SD3B3_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B3_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet4()
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(7, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier_voxel = nn.Sequential(nn.Linear(256, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        # print("*****************", x.F.shape)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        # x4 = self.net.layer4(x3)
        # # x4 = self.process(x4)
        z1 = voxel_to_point(x3, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out

class SD3B_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet50(pretrained=False)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        # print("*****************", x.F.shape)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: scale factor
        Return:
            self-attention
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # torch.Size([16, 149, 149])
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)

        context = torch.matmul(attention, V)  # torch.Size([16, 149, 64])

        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.fc_s = nn.Linear(num_head * self.dim_head, dim_model)
        self.fc_i = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.batch_norm = nn.BatchNorm1d(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        # out = self.layer_norm(out)
        out = self.batch_norm(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.1):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.batch_norm = nn.BatchNorm1d(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # residual connection
        out = self.batch_norm(out.permute(0, 2, 1))
        out = out.permute(0, 2, 1)
        return out

class Trans_ResNet50Merge(nn.Module):

    def __init__(self, **kwargs):
        super(Trans_ResNet50Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        encoder_list = [
            Encoder(dim_model=512, num_head=8, hidden=1024, dropout=0.3)
            for _ in range(1)]

        self.net = nn.Sequential(*encoder_list)
        self.fc_process = nn.Conv1d(7, 512, kernel_size=11, padding=5)
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z1 = self.fc_process(x.permute(0, 2, 1))
        z1 = z1.permute(0, 2, 1)
        z1 = self.net(z1)
        pooled_data = torch.mean(z1, dim=1)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data


class Trans_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(Trans_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        encoder_list = [
            Encoder(dim_model=512, num_head=8, hidden=1024, dropout=0.3)
            for _ in range(1)]

        self.net = nn.Sequential(*encoder_list)
        self.fc_process = nn.Conv1d(9, 512, kernel_size=11, padding=5)
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z1 = self.fc_process(x.permute(0, 2, 1))
        z1 = z1.permute(0, 2, 1)
        z1 = self.net(z1)
        pooled_data = torch.mean(z1, dim=1)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out


class GNN_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(GNN_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        # encoder_list = [
        #     Encoder(dim_model=512, num_head=8, hidden=1024, dropout=0.3)
        #     for _ in range(1)]

        self.net = GNN_module(nway=128, input_dim=64,
            hidden_dim=128,
            num_layers=1,
            feature_type='dense')
        self.fc_process = nn.Conv1d(9, 64, kernel_size=11, padding=5)
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.classifier_voxel = nn.Sequential(nn.Linear(128, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z1 = self.fc_process(x.permute(0, 2, 1))
        z1 = z1.permute(0, 2, 1)
        z1 = self.net(z1)
        pooled_data = self.classifier_voxel(z1)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out

class RNN_ResNet50Merge(nn.Module):

    def __init__(self, **kwargs):
        super(RNN_ResNet50Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = nn.Sequential(nn.GRU(input_size=7, hidden_size=256, num_layers=4, batch_first=True,
                                             dropout=0.5, bidirectional=True))
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z1, _ = self.net(x)
        # print(z1.shape)
        pooled_data = torch.mean(z1, dim=1)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data

class RNN_ResNet50Norm(nn.Module):

    def __init__(self, **kwargs):
        super(RNN_ResNet50Norm, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = nn.Sequential(nn.GRU(input_size=9, hidden_size=256, num_layers=4, batch_first=True,
                                             dropout=0.5, bidirectional=True))
        self.img_branch = resnet2d.resnet50(pretrained=True)

        self.classifier_voxel = nn.Sequential(nn.Linear(512, 2048),
                                              nn.Hardswish(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z1, _ = self.net(x)
        # print(z1.shape)
        pooled_data = torch.mean(z1, dim=1)
        pooled_data = self.classifier_voxel(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out
class SD3B_ResNet50Solo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet50Solo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.img_branch = resnet2d.resnet50(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out
class SD3B_SwinMerge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_SwinMerge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = swin.SwinTransformerV2(img_size=256, patch_size=4, in_chans=3, num_classes=1000, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32], window_size=16, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
                          drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False,
                          pretrained_window_sizes=[12, 12, 12, 6])
        ckpt = torch.load("/media/kemove/403/yangjingru/spvnasaux/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth")
        self.img_branch.load_state_dict(ckpt["model"], strict=False)
        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Linear(512, 1024)
        self.classifier = nn.Sequential(nn.Linear(1024, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        ### first interaction
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)

        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier1(pooled_data)
        merge_out = img_out + pooled_data
        out = self.classifier(merge_out)
        return out, img_out, pooled_data

class SD3B_SwinSolo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_SwinSolo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        # self.net = resnet18()
        self.img_branch = swin.SwinTransformerV2(img_size=256, patch_size=4, in_chans=3, num_classes=1000, embed_dim=128, depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32], window_size=16, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0,
                          drop_path_rate=0.2, ape=False, patch_norm=True, use_checkpoint=False,
                          pretrained_window_sizes=[12, 12, 12, 6])
        ckpt = torch.load("/media/kemove/403/yangjingru/spvnasaux/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth")
        self.img_branch.load_state_dict(ckpt["model"], strict=False)
        self.classifier = nn.Sequential(nn.Linear(1024, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out #, img_out, pooled_data


class SD3B_ResNet101Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet101Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet101(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 2048))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier1(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data

class SD3B_ResNet101Solo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet101Solo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.img_branch = resnet2d.resnet101(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out


class SD3B_ResNet152Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet152Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = resnet2d.resnet152(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 2048))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        pooled_data = self.classifier1(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data

class SD3B_ResNet152Solo(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_ResNet152Solo, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.img_branch = resnet2d.resnet152(pretrained=True)
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        img_out = self.img_branch(img)
        out = self.classifier(img_out)
        return out

class SD3B_Mbv2Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_Mbv2Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = mbv2.mobilenet_v2(pretrained=True)
        # self.img_branch_postprocess = resnet2d.conv1x1(960, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 1280))
        self.classifier = nn.Sequential(nn.Linear(1280, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = self.classifier1(pooled_data) + img_out
        out = self.classifier(merge_out)
        return out


class SD3B_Incv3Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_Incv3Merge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = inv3.inception_v3(pretrained=True)
        # self.img_branch_postprocess = resnet2d.conv1x1(960, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 2048))
        self.classifier = nn.Sequential(nn.Linear(2048, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = self.classifier1(pooled_data) + img_out[0]
        out = self.classifier(merge_out)
        return out


class SD3B_DenseMerge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_DenseMerge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = densenet.densenet169(pretrained=True)
        # self.img_branch_postprocess = resnet2d.conv1x1(960, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 1664))
        self.classifier = nn.Sequential(nn.Linear(1664, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = self.classifier1(pooled_data) + img_out
        out = self.classifier(merge_out)
        return out


class SD3B_Mbv3Merge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_Mbv3Merge, self).__init__()

        self.pres = 1.0
        self.vres = 1.0

        self.net = resnet18()
        self.img_branch = mobilenet.mobilenet_v3_large(pretrained=True, width_mult=1.0, reduced_tail=False, dilated=False)
        # self.img_branch_postprocess = resnet2d.conv1x1(960, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        # self.classifier3 = nn.Sequential(nn.Linear(256, 512))
        # self.classifier2 = nn.Sequential(nn.Linear(128, 512))
        self.classifier1 = nn.Sequential(nn.Linear(512, 1280))
        self.classifier = nn.Sequential(nn.Linear(1280, 345))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        # pooled_data2 = torch_scatter.scatter_mean(x2.F, x2.C[:, -1].long(), dim=0)
        x3 = self.net.layer3(x2)
        # pooled_data3 = torch_scatter.scatter_mean(x3.F, x3.C[:, -1].long(), dim=0)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        # pooled_data3 = self.classifier3(pooled_data3)
        # pooled_data2 = self.classifier2(pooled_data2)
        # pooled_data = pooled_data + pooled_data3
        pooled_data = self.classifier1(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data


class SD3B_Mbv3HMerge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_Mbv3HMerge, self).__init__()

        self.pres = 1.0
        self.vres = 1.0

        self.net = resnet18()
        self.img_branch = mobilenetnv3.mobilenet_v3_large(pretrained=True, width_mult=1.0, reduced_tail=False,
                                                       dilated=False)
        # self.img_branch_postprocess = resnet2d.conv1x1(960, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        # self.classifier3 = nn.Sequential(nn.Linear(256, 512))
        # self.classifier2 = nn.Sequential(nn.Linear(128, 512))
        self.classifier1 = nn.Sequential(nn.Linear(512, 1280))
        self.classifier = nn.Sequential(nn.Linear(1280, 345))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        # pooled_data2 = torch_scatter.scatter_mean(x2.F, x2.C[:, -1].long(), dim=0)
        x3 = self.net.layer3(x2)
        # pooled_data3 = torch_scatter.scatter_mean(x3.F, x3.C[:, -1].long(), dim=0)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        # pooled_data3 = self.classifier3(pooled_data3)
        # pooled_data2 = self.classifier2(pooled_data2)
        # pooled_data = pooled_data + pooled_data3
        pooled_data = self.classifier1(pooled_data)
        merge_out = pooled_data + img_out
        out = self.classifier(merge_out)
        return out, img_out, pooled_data


class SD3B_EffMerge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_EffMerge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = efficientnet.efficientnet_b4(pretrained=True)
        self.img_branch_postprocess = resnet2d.conv1x1(1792, 512)
        # self.process = BasicConvolutionBlock(2048, 512)

        self.stem = nn.Sequential(
            spnn.Conv3d(6, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))

        self.classifier = nn.Sequential(nn.Linear(512, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch_postprocess(self.img_branch(img))
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = pooled_data + img_out.squeeze(dim=-1).squeeze(dim=-1)
        out = self.classifier(merge_out)
        return out


class SD3B_NasMerge(nn.Module):

    def __init__(self, **kwargs):
        super(SD3B_NasMerge, self).__init__()

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.net = resnet18()
        self.img_branch = mnasnet.mnasnet1_0(pretrained=True)

        self.stem = nn.Sequential(
            spnn.Conv3d(9, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True),
            spnn.Conv3d(64, 64, kernel_size=3, stride=1),
            spnn.BatchNorm(64), spnn.ReLU(True))
        self.classifier1 = nn.Sequential(nn.Linear(512, 1280))
        self.classifier = nn.Sequential(nn.Linear(1280, kwargs['num_classes']))
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img):
        img_out = self.img_branch(img)
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.net.layer1(x1)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        # x4 = self.process(x4)
        z1 = voxel_to_point(x4, z0)
        pooled_data = torch_scatter.scatter_mean(z1.F, z1.C[:, -1].long(), dim=0)
        merge_out = self.classifier1(pooled_data) + img_out
        out = self.classifier(merge_out)
        return out



