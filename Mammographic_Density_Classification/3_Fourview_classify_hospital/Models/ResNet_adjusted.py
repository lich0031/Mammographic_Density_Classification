import torch.nn as nn
import os
import torch
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)

        x3 = self.layer1(x3)
        x3 = self.layer2(x3)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)

        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)

        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)

        x4 = self.layer1(x4)
        x4 = self.layer2(x4)
        x4 = self.layer3(x4)
        x4 = self.layer4(x4)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ResNet_Noshare(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet_Noshare, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 64
        self.view1_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.view1_bn1 = nn.BatchNorm2d(64)
        self.view1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.view1_layer1 = self._make_layer(block, 64, layers[0])
        self.view1_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.view1_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.view1_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 64
        self.view2_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.view2_bn1 = nn.BatchNorm2d(64)
        self.view2_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.view2_layer1 = self._make_layer(block, 64, layers[0])
        self.view2_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.view2_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.view2_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 64
        self.view3_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.view3_bn1 = nn.BatchNorm2d(64)
        self.view3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.view3_layer1 = self._make_layer(block, 64, layers[0])
        self.view3_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.view3_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.view3_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.inplanes = 64
        self.view4_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.view4_bn1 = nn.BatchNorm2d(64)
        self.view4_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.view4_layer1 = self._make_layer(block, 64, layers[0])
        self.view4_layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.view4_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.view4_layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        x1 = self.view1_conv1(x1)
        x1 = self.view1_bn1(x1)
        x1 = self.relu(x1)
        x1 = self.view1_maxpool(x1)

        x1 = self.view1_layer1(x1)
        x1 = self.view1_layer2(x1)
        x1 = self.view1_layer3(x1)
        x1 = self.view1_layer4(x1)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.view2_conv1(x2)
        x2 = self.view2_bn1(x2)
        x2 = self.relu(x2)
        x2 = self.view2_maxpool(x2)

        x2 = self.view2_layer1(x2)
        x2 = self.view2_layer2(x2)
        x2 = self.view2_layer3(x2)
        x2 = self.view2_layer4(x2)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.view3_conv1(x3)
        x3 = self.view3_bn1(x3)
        x3 = self.relu(x3)
        x3 = self.view3_maxpool(x3)

        x3 = self.view3_layer1(x3)
        x3 = self.view3_layer2(x3)
        x3 = self.view3_layer3(x3)
        x3 = self.view3_layer4(x3)

        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)

        x4 = self.view4_conv1(x4)
        x4 = self.view4_bn1(x4)
        x4 = self.relu(x4)
        x4 = self.view4_maxpool(x4)

        x4 = self.view4_layer1(x4)
        x4 = self.view4_layer2(x4)
        x4 = self.view4_layer3(x4)
        x4 = self.view4_layer4(x4)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def resnet18(num_classes, pretrain):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrain:
        model = models.resnet18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def resnet34(num_classes, pretrain):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrain:
        model = models.resnet34(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50(num_classes, pretrain):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain:
        model = models.resnet50(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50noshare(num_classes, pretrain):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain:
        model = models.resnet50(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet_Noshare(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet101(num_classes, pretrain):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain:
        model = models.resnet101(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

def resnet152(num_classes, pretrain):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrain:
        model = models.resnet152(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = resnet50noshare(num_classes=4, pretrain=False).cuda()
    print(net)

    test_x = Variable(torch.ones(4, 3, 224, 224))
    out_x = net(test_x.cuda(), test_x.cuda(), test_x.cuda(), test_x.cuda())

    print(out_x)
    print(out_x.shape)