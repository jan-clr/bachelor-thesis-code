import torch
import torch.nn as nn
import timm
import itertools
from torchvision import models
from typing import List
from torch.nn import functional as F
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """
    Performs two same convolutions back to back, accepting an input with in_ch channels and outputting out_ch channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        """

        :param out_ch: nr of desired output channels
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True)
        )
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


class CS_UNET(nn.Module):
    """
    Basic UNET architecture to learn on the Cityscape Dataset
    """

    def __init__(self, in_ch=3, out_ch=2, feature_steps=[64, 128, 256, 512]):
        super(CS_UNET, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        # define contracting path
        for features in feature_steps:
            self.down.append(DoubleConv(in_ch=in_ch, out_ch=features))
            in_ch = features

        # define expansive path
        for features in reversed(feature_steps):
            self.up.append(nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2))
            self.up.append(DoubleConv(features * 2, features))

        self.bottom = DoubleConv(feature_steps[-1], feature_steps[-1] * 2)
        self.final = nn.Conv2d(feature_steps[0], out_ch, kernel_size=1)

    def forward(self, x):
        x_down = []

        for i, down_step in enumerate(self.down):
            x = down_step(x)
            x_down.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        x_down = x_down[::-1]

        for i, up_step in enumerate(pairwise_iter(self.up)):
            # perform ConvTransposed
            x = up_step[0](x)
            # concat downwards result
            x = torch.cat((x_down[i], x), dim=1)
            # perform DoubleConv
            x = up_step[1](x)

        return self.final(x)


class UnetResEncoder(nn.Module):
    """
    UNET Implementation using pretrained ResNet as Encoder
    """

    def __init__(self, in_ch=3, out_ch=2, encoder_name='resnet34', freeze_encoder=False, dropout_p=None, out_indices=None):
        super(UnetResEncoder, self).__init__()

        if dropout_p is not None:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True, in_chans=in_ch,
                                         drop_rate=dropout_p or 0.0, out_indices=out_indices)
        feature_steps = self.encoder.feature_info.channels()
        self.up = nn.ModuleList()

        # define expansive path
        for features in itertools.pairwise(reversed(feature_steps)):
            self.up.append(nn.ConvTranspose2d(features[0], features[1], kernel_size=2, stride=2))
            self.up.append(DoubleConv(features[1] * 2, features[1]))

        self.decode_final = nn.Sequential(
            nn.ConvTranspose2d(feature_steps[0], feature_steps[0], kernel_size=2, stride=2),
            DoubleConv(feature_steps[0], feature_steps[0])
        )

        self.final = nn.Conv2d(feature_steps[0], out_ch, kernel_size=1)

        self.init_weights()
        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, x):
        out_down = self.encoder(x)

        out_down = out_down[::-1]
        x = out_down[0]

        for i, up_step in enumerate(pairwise_iter(self.up)):
            # perform ConvTransposed
            x = up_step[0](x)
            # concat downwards result
            x = torch.cat((out_down[i + 1], x), dim=1)
            if self.dropout is not None:
                x = self.dropout(x)
            # perform DoubleConv
            x = up_step[1](x)

        x = self.decode_final(x)

        return self.final(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)

        nn.init.xavier_normal_(self.final.weight)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


class ASPPConv(nn.Sequential):
    """
    Taken from
    https://github.com/ChristmasFan/SSL_Denoising_Segmentation
    https://arxiv.org/abs/2210.10426
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    Taken from
    https://github.com/ChristmasFan/SSL_Denoising_Segmentation
    https://arxiv.org/abs/2210.10426
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Taken from
    https://github.com/ChristmasFan/SSL_Denoising_Segmentation
    https://arxiv.org/abs/2210.10426
    """
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3plus(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, encoder_name='resnet101', dropout_p=None):
        super(DeepLabV3plus, self).__init__()
        self.backbone = timm.create_model(encoder_name, pretrained=True, features_only=True, in_chans=in_ch,
                                          drop_rate=dropout_p or 0.0, output_stride=16)
        low_level_features = 256
        self.aspp = ASPP(2048, [6, 12, 18], out_channels=256)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(in_channels=low_level_features, out_channels=48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        low_level_features = self.low_level_project(features[1])
        high_level_features = self.aspp(features[-1])
        high_level_features = F.interpolate(high_level_features, size=low_level_features.shape[2:], mode='bilinear',
                                            align_corners=False)

        return F.interpolate(self.classifier(torch.cat([low_level_features, high_level_features], dim=1)),
                             x.size()[-2:], mode='bilinear', align_corners=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


def pairwise_iter(iterable):
    """
    | Return an iterator that returns the elements of an iterable pairwise
    | s -> (s0, s1), (s2, s3), (s4, s5), ...
    | Credit to [Pairwise]_
    :param iterable: iterable s = {s0, s1, s2, ...}
    :return: Iterator that returns (s0, s1), (s2, s3), (s4, s5), ...
    .. [Pairwise] https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
    """
    a = iter(iterable)
    return zip(a, a)


def test_deeplabv3p():
    model = smp.DeepLabV3Plus(in_channels=3, classes=19, encoder_name='resnet101', encoder_weights='imagenet')
    model1 = DeepLabV3plus(in_ch=3, num_classes=19)
    print(model1)
    print(model)
    i, j = 0, 0
    for m in model1.backbone.modules():
        i += 1
    for m in model.encoder.modules():
        j += 1
    print(i, j)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape)


def test():
    x = torch.randn((4, 3, 512, 256))
    model = UnetResEncoder(out_indices=(0, 1, 2))
    model.eval()
    out = model(x)
    for o in out:
        print(o.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    test()


if __name__ == '__main__':
    main()
