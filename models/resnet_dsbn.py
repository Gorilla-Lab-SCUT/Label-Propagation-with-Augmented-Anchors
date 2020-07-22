import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import ipdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.bn1_temp = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_temp = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3_temp = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inp):
        x, ST = inp[0], inp[1]

        residual = x

        out = self.conv1(x)
        if ST == 'S':
            out = self.bn1(out)
        else:
            out = self.bn1_temp(out)
        out = self.relu(out)

        out = self.conv2(out)
        if ST == 'S':
            out = self.bn2(out)
        else:
            out = self.bn2_temp(out)
        out = self.relu(out)

        out = self.conv3(out)
        if ST == 'S':
            out = self.bn3(out)
        else:
            out = self.bn3_temp(out)

        if self.downsample is not None:
            residual = self.downsample(x, ST)

        out += residual
        out = self.relu(out)
        ################################

        return {0:out, 1:ST}

class DownSample(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride =1, bias=False):
        super(DownSample, self).__init__()
        self.downconv = nn.Conv2d(inplanes, planes,
                           kernel_size=kernel_size, stride=stride, bias=False)
        self.downbn = nn.BatchNorm2d(planes)
        self.downbn_temp = nn.BatchNorm2d(planes)

    def forward(self, x, ST):
        x = self.downconv(x)
        if ST == 'S':
            out = self.downbn(x)
        else:
            out = self.downbn_temp(x)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_temp = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownSample(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, ST='S'):
        x = self.conv1(x)
        if ST == 'S':
            x = self.bn1(x)
        else:
            x = self.bn1_temp(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1({0:x, 1:ST})
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        ##################################### feature s stream , out s stream, feature t stream, out t stream.
        return x, out


def resnet18(args, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)
    model.fc.weight.data.normal_(0.0, 0.02)
    model.fc.bias.data.normal_(0)
    return model


def resnet34(args, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        print('Load ImageNet pre-trained resnet model')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)

    return model


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        if args.pretrained_checkpoint:
            # modify the structure of the model.
            print('load the source data pretrained model from: ', args.pretrained_checkpoint)
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
            init_dict = model.state_dict()
            pretrained_dict = torch.load(args.pretrained_checkpoint)['state_dict']
            pretrained_dict_temp = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            for k, v in init_dict.items():
                if k in pretrained_dict_temp.keys():  ############## the key exist in the dictionary
                    init_dict[k] = pretrained_dict_temp[k]
                else:  ############## the resuired key not exist in the dict
                    if k.find('num_batches_tracked') != -1:  ######### skip
                        print(
                            k)  ##################################### all the parameters are updated using the pretrained para, except the printed ones.
                    elif k.find('downsample.downconv') != -1:
                        pretrained_key = k.replace('downsample.downconv', 'downsample.0')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn_temp') != -1:
                        pretrained_key = k.replace('downsample.downbn_temp', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn') != -1:
                        pretrained_key = k.replace('downsample.downbn', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('temp') != -1:  ######### copy the pretrained bn
                        pretrained_key = k.replace('_temp', '')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    else:
                        ipdb.set_trace()


            # pretrained_dict.pop('fc.weight')
            # pretrained_dict.pop('fc.bias')
            # init_dict.update(pretrained_dict_temp)
            model.load_state_dict(init_dict)
        else:
            print('load the imagenet pretrained model')
            pretrained_dict_temp = model_zoo.load_url(model_urls['resnet50'])
            init_dict = model.state_dict()
            for k, v in init_dict.items():
                if k in pretrained_dict_temp.keys(): ############## the key exist in the dictionary
                    init_dict[k] = pretrained_dict_temp[k]
                else:                                ############## the resuired key not exist in the dict
                    if k.find('num_batches_tracked') != -1: ######### skip
                        print(k)   ##################################### all the parameters are updated using the pretrained para, except the printed ones.
                    elif k.find('downsample.downconv') != -1:
                        pretrained_key = k.replace('downsample.downconv', 'downsample.0')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn_temp') != -1:
                        pretrained_key = k.replace('downsample.downbn_temp', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn') != -1:
                        pretrained_key = k.replace('downsample.downbn', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('temp') != -1:              ######### copy the pretrained bn
                        pretrained_key = k.replace('_temp', '')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    else:
                        ipdb.set_trace()

            model.load_state_dict(init_dict)  ################# load this directly to confirm all the keys are updated
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
    return model


def resnet101(args, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if args.pretrained:
        if args.pretrained_checkpoint:
            # modify the structure of the model.
            print('load the source data pretrained model from: ', args.pretrained_checkpoint)
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
            init_dict = model.state_dict()
            pretrained_dict_temp = torch.load(args.pretrained_checkpoint)['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict_temp.items()}
            pretrained_dict.pop('fc.weight')
            pretrained_dict.pop('fc.bias')
            init_dict.update(pretrained_dict)
            model.load_state_dict(init_dict)
        else:
            print('load the imagenet pretrained model')
            pretrained_dict_temp = model_zoo.load_url(model_urls['resnet101'])
            init_dict = model.state_dict()
            for k, v in init_dict.items():
                if k in pretrained_dict_temp.keys(): ############## the key exist in the dictionary
                    init_dict[k] = pretrained_dict_temp[k]
                else:                                ############## the resuired key not exist in the dict
                    if k.find('num_batches_tracked') != -1: ######### skip
                        print(k)   ##################################### all the parameters are updated using the pretrained para, except the printed ones.
                    elif k.find('downsample.downconv') != -1:
                        pretrained_key = k.replace('downsample.downconv', 'downsample.0')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn_temp') != -1:
                        pretrained_key = k.replace('downsample.downbn_temp', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('downsample.downbn') != -1:
                        pretrained_key = k.replace('downsample.downbn', 'downsample.1')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    elif k.find('temp') != -1:              ######### copy the pretrained bn
                        pretrained_key = k.replace('_temp', '')
                        if pretrained_key in pretrained_dict_temp.keys():
                            init_dict[k] = pretrained_dict_temp[pretrained_key]
                        else:
                            ipdb.set_trace()
                    else:
                        ipdb.set_trace()

            model.load_state_dict(init_dict)  ################# load this directly to confirm all the keys are updated
            num_of_feature_map = model.fc.in_features
            model.fc = nn.Linear(num_of_feature_map, args.num_classes * 2)
    return model

def resnet152(args, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if args.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    # modify the structure of the model.
    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)
    
    return model


def resnet(args, **kwargs):
    print("==> creating model '{}' ".format(args.arch))
    if args.arch == 'resnet18':
        return resnet18(args)
    elif args.arch == 'resnet34':
        return resnet34(args)
    elif args.arch == 'resnet50':
        return resnet50(args)
    elif args.arch == 'resnet101':
        return resnet101(args)
    elif args.arch == 'resnet152':
        return resnet152(args)
    else:
        raise ValueError('Unrecognized model architecture', args.arch)