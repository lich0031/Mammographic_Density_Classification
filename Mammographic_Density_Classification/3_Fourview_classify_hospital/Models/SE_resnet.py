import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import os
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y + x

class Spatial_Attention(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Spatial_Attention, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, input_channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv3 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation,
                               stride=1, padding=dilation)
        self.conv4 = nn.Conv2d(input_channel // reduction, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn(self.conv4(y))
        y = self.sigmoid(y)

        return x * y + x

class Bottleneck_Attention_Module(nn.Module):
    def __init__(self, input_channel, reduction=16, dilation=4):
        super(Bottleneck_Attention_Module, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(input_channel, input_channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(input_channel // reduction, input_channel),
                nn.Sigmoid()
        )

        self.conv1 = nn.Conv2d(input_channel, input_channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation, stride=1, padding=dilation)
        self.conv3 = nn.Conv2d(input_channel // reduction, input_channel // reduction, kernel_size=3, dilation=dilation, stride=1, padding=dilation)
        self.conv4 = nn.Conv2d(input_channel // reduction, 1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        ca_weights = torch.ones(x.size()).cuda() * y1

        y2 = self.conv1(x)
        y2 = self.conv2(y2)
        y2 = self.conv3(y2)
        y2 = self.bn(self.conv4(y2))

        sa_weights = y2.repeat(1, x.size()[1], 1, 1)

        y = self.sigmoid(ca_weights + sa_weights)

        return x * y + x

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
        self.se1 = Channel_Attention(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.se2 = Channel_Attention(64 * block.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.se3 = Channel_Attention(128 * block.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.se4 = Channel_Attention(256 * block.expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.se5 = Channel_Attention(512 * block.expansion)
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
        x1 = self.se1(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.se2(x1)
        x1 = self.layer2(x1)
        x1 = self.se3(x1)
        x1 = self.layer3(x1)
        x1 = self.se4(x1)
        x1 = self.layer4(x1)
        x1 = self.se5(x1)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.se1(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.se2(x2)
        x2 = self.layer2(x2)
        x2 = self.se3(x2)
        x2 = self.layer3(x2)
        x2 = self.se4(x2)
        x2 = self.layer4(x2)
        x2 = self.se5(x2)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.se1(x3)
        x3 = self.maxpool(x3)

        x3 = self.layer1(x3)
        x3 = self.se2(x3)
        x3 = self.layer2(x3)
        x3 = self.se3(x3)
        x3 = self.layer3(x3)
        x3 = self.se4(x3)
        x3 = self.layer4(x3)
        x3 = self.se5(x3)

        x3 = self.avgpool(x3)
        x3 = x3.view(x3.size(0), -1)

        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        x4 = self.se1(x4)
        x4 = self.maxpool(x4)

        x4 = self.layer1(x4)
        x4 = self.se2(x4)
        x4 = self.layer2(x4)
        x4 = self.se3(x4)
        x4 = self.layer3(x4)
        x4 = self.se4(x4)
        x4 = self.layer4(x4)
        x4 = self.se5(x4)

        x4 = self.avgpool(x4)
        x4 = x4.view(x4.size(0), -1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def resnet18se(num_classes, pretrain):
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


def resnet34se(num_classes, pretrain):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrain:
        model = models.resnet18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, num_classes)
    else:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50se(num_classes, pretrain):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    if pretrain:
        # model = models.resnet50(pretrained=True)
        # fc_features = model.fc.in_features
        # model.fc = nn.Linear(fc_features, num_classes)
        pass

    return model

def resnet101se(num_classes, pretrain):
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

def resnet152se(num_classes, pretrain):
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
    net = resnet18se(num_classes=4, pretrain=False).cuda()
    print(net)

    test_x = Variable(torch.ones(4, 3, 224, 224))
    out_x = net(test_x.cuda(), test_x.cuda(), test_x.cuda(), test_x.cuda())

    print(out_x)
    print(out_x.shape)