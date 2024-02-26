import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_pretrained_url(key):
    url_prefix = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{url_prefix}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )

        print(verbose)
    return model


def always_evaluate(model_method):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.eval()
        with torch.no_grad():
            return model_method(*args, **kwargs)
    return wrapper


def remove_prefix(state_dict, prefix):
    return {key.replace(prefix, ""): value for key, value in state_dict.items() if key.startswith(prefix)}


class ResNet50(nn.Module):
    def __init__(self, path):
        super(ResNet50, self).__init__()

        self.model = resnet50(pretrained=False, progress=False, key=None)
        self.model.load_state_dict(remove_prefix(torch.load(path)['state_dict'], 'model.'))

        for param in self.model.parameters():
            param.requires_grad = False

        self.extractor = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten()
                                       )

    @always_evaluate
    def forward(self, x):
        out = self.model(x)
        out = self.extractor(out)

        return out


if __name__ == '__main__':
    from torchsummary import summary
    # import torch.nn.functional as F

    model = ResNet50(path='../simclr_pretrained_model_ckpt/checkpoint_0200_Adam.pt').cuda()
    summary(model, (3, 224, 224))
