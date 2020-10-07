import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.drop import DropBlock2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop=False, block_size=4):
        super(BasicBlock, self).__init__()
        self.drop = drop
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if drop:
            self.drop1 = DropBlock2d(block_size=block_size)
        self.conv2 = conv3x3(planes, planes)
        if drop:
            self.bn2 = nn.BatchNorm2d(planes)
        self.drop2 = DropBlock2d(block_size=block_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.drop:
            out = self.drop1(F.relu(self.bn1(self.conv1(x))))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.drop2(F.relu(out))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop=False, block_size=4):
        super(PreActBlock, self).__init__()
        self.drop = drop
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        if drop:
            self.drop1 = DropBlock2d(block_size=block_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        if drop:
            self.drop2 = DropBlock2d(block_size=block_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.drop:
            out = self.drop1(F.relu(self.bn1(x)))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
            out = self.drop2(out)
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, drop=False, block_size=4):
        super(Bottleneck, self).__init__()
        self.drop = drop
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if drop:
            self.drop1 = DropBlock2d(block_size=block_size)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if drop:
            self.drop2 = DropBlock2d(block_size=block_size)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        if drop:
            self.drop3 = DropBlock2d(block_size=block_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.drop:
            out = self.drop1(F.relu(self.bn1(self.conv1(x))))
            out = self.drop2(F.relu(self.bn2(self.conv2(out))))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = self.drop3(F.relu(out))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, drop=False, block_size=4):
        super(PreActBottleneck, self).__init__()
        self.drop = drop
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if drop:
            self.drop1 = DropBlock2d(block_size=block_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if drop:
            self.drop2 = DropBlock2d(block_size=block_size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if drop:
            self.drop3 = DropBlock2d(block_size=block_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        if self.drop:
            out = self.drop1(F.relu(self.bn1(x)))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(self.drop2(F.relu(self.bn2(out))))
            out = self.conv3(F.relu(self.bn3(out)))
            out += shortcut
            out = self.drop3(out)
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out)
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))
            out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, drop=False):
        super(ResNet, self).__init__()
        self.drop = drop
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop=drop, block_size=5)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop=drop, block_size=3)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, drop=False, block_size=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop=drop, block_size=block_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        return out

    def classifier(self, x, lin=0, lout=5):
        out = x
        if lout > 4:
            out = self.linear(out)
        return out

    def forward(self, x, lin=0, lout=5):
        out = self.features(x, lin, lout)
        out = self.classifier(out, lin, lout)
        return out


def ResNet18(num_classes=10, **kwargs):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def ResNet34(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def ResNet50(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def ResNet101(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def ResNet152(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
