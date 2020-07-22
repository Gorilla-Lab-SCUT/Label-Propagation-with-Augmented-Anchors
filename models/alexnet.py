import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import ipdb


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.fc(x)
        return x


def alexnet(args, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if args.pretrained:
        print('Load ImageNet pre-trained alexnet model')
        pretrained_state_dict = model_zoo.load_url(model_urls['alexnet'])
        pretrained_state_dict['features1.0.weight'] = pretrained_state_dict.pop('features.0.weight')
        pretrained_state_dict['features1.0.bias'] = pretrained_state_dict.pop('features.0.bias')
        pretrained_state_dict['features1.3.weight'] = pretrained_state_dict.pop('features.3.weight')
        pretrained_state_dict['features1.3.bias'] = pretrained_state_dict.pop('features.3.bias')
        pretrained_state_dict['features1.6.weight'] = pretrained_state_dict.pop('features.6.weight')
        pretrained_state_dict['features1.6.bias'] = pretrained_state_dict.pop('features.6.bias')
        pretrained_state_dict['features2.0.weight'] = pretrained_state_dict.pop('features.8.weight')
        pretrained_state_dict['features2.0.bias'] = pretrained_state_dict.pop('features.8.bias')
        pretrained_state_dict['features2.2.weight'] = pretrained_state_dict.pop('features.10.weight')
        pretrained_state_dict['features2.2.bias'] = pretrained_state_dict.pop('features.10.bias')
        pretrained_state_dict['fc.weight'] = pretrained_state_dict.pop('classifier.6.weight')
        pretrained_state_dict['fc.bias'] = pretrained_state_dict.pop('classifier.6.bias')
        model.load_state_dict(pretrained_state_dict)
    # modify the structure of the model.
    # cf. https://github.com/meliketoy/fine-tuning.pytorch/blob/master/main.py
    # if (args.net_type == 'alexnet' or args.net_type == 'vggnet'):
    #     num_ftrs = model_ft.classifier[6].in_features
    #     feature_model = list(model_ft.classifier.children())
    #     feature_model.pop()
    #     feature_model.append(nn.Linear(num_ftrs, len(dset_classes)))
    #     model_ft.classifier = nn.Sequential(*feature_model)
    # elif (args.net_type == 'resnet'):
    #     num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Linear(num_ftrs, len(dset_classes))

    num_of_feature_map = model.fc.in_features
    model.fc = nn.Linear(num_of_feature_map, args.num_classes)
    return model