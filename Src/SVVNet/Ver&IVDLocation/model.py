import torch
from torch import nn
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool3d
from torch.nn import ReLU, Sigmoid, Sequential, Tanh, Upsample
from tensorboardX import SummaryWriter
import torch

"""
The Unet.
from paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""


class Conv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, residual=None):
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(in_channels, out_channels, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_channels),
            ReLU()
        )

        self.conv2 = Sequential(
            Conv3d(out_channels, out_channels, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_channels),
            ReLU()
        )
        self.residual = residual
        if self.residual is not None:
            self.residual_upsampler = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()
        self.deconv = Sequential(
            ConvTranspose3d(in_channels, out_channels, kernel_size=(1, kernel, kernel),
                            stride=(1, stride, stride), padding=(0, padding, padding), output_padding=0,
                            bias=True),
            ReLU()
        )

    def forward(self, x):
        return self.deconv(x)


class UNet3D(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, residual='conv', filters_base=64, *args, **kwargs):
        super(UNet3D, self).__init__()

        filters = [filters_base, 2 * filters_base, 4 * filters_base, 8 * filters_base, 16 * filters_base]

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1, 2, 2))
        self.pool2 = MaxPool3d((1, 2, 2))
        self.pool3 = MaxPool3d((1, 2, 2))
        self.pool4 = MaxPool3d((1, 2, 2))

        # Encoder convolutions
        self.conv1 = Conv3D_Block(in_channels, filters[0], residual=residual)
        self.conv2 = Conv3D_Block(filters[0], filters[1], residual=residual)
        self.conv3 = Conv3D_Block(filters[1], filters[2], residual=residual)
        self.conv4 = Conv3D_Block(filters[2], filters[3], residual=residual)
        self.conv5 = Conv3D_Block(filters[3], filters[4], residual=residual)

        # Decoder
        self.up5 = Deconv3D_Block(filters[4], filters[3])
        self.up5_conv = Conv3D_Block(2 * filters[3], filters[3], residual=residual)
        self.up4 = Deconv3D_Block(filters[3], filters[2])
        self.up4_conv = Conv3D_Block(2 * filters[2], filters[2], residual=residual)
        self.up3 = Deconv3D_Block(filters[2], filters[1])
        self.up3_conv = Conv3D_Block(2 * filters[1], filters[1], residual=residual)
        self.up2 = Deconv3D_Block(filters[1], filters[0])
        self.up2_conv = Conv3D_Block(2 * filters[0], filters[0], residual=residual)

        # Final output
        # self.Sigmoid = Sigmoid()
        self.convFinal = Conv3d(filters[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        # encoder part
        e1 = self.conv1(input)

        e2 = self.pool1(e1)
        e2 = self.conv2(e2)

        e3 = self.pool2(e2)
        e3 = self.conv3(e3)

        e4 = self.pool3(e3)
        e4 = self.conv4(e4)

        e5 = self.pool4(e4)
        e5 = self.conv5(e5)

        # decoder part
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.up5_conv(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.up4_conv(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.up3_conv(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.up2_conv(d2)

        # output = self.Sigmoid(self.convFinal(d2))
        output = self.convFinal(d2)

        return output


"""
The SpatialConfigurationNet.
from paper "Coarse to Fine Vertebrae Localization and Segmentation with SpatialConfiguration-Net and U-Net"
"""


class SpatialConfigurationNet(nn.Module):
    def __init__(self,
                 num_labels,
                 num_filters_base=64,
                 spatial_downsample=4,
                 residual='conv',
                 channels=1):
        """
        Initializer.
        :param num_labels: Number of outputs.
        :param num_filters_base: Number of filters for the local appearance and spatial configuration sub-networks.
        :param spatial_downsample: Downsample factor for spatial configuration output.
        """
        super(SpatialConfigurationNet, self).__init__()

        self.num_filters_base = num_filters_base

        # 下采样率
        self.downsampling_factor = spatial_downsample

        # local part
        self.scnet_local = UNet3D(in_channels=channels, num_classes=num_labels, filters=self.num_filters_base)

        # spatial part
        self.downsampling = AvgPool3d((1, self.downsampling_factor, self.downsampling_factor))

        # verse 2019 conv x4
        self.conv1 = Sequential(
            Conv3d(in_channels=num_labels, out_channels=num_labels, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            ReLU()
        )
        self.conv2 = Sequential(
            Conv3d(in_channels=num_labels, out_channels=num_labels, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            ReLU()
        )
        self.conv3 = Sequential(
            Conv3d(in_channels=num_labels, out_channels=num_labels, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            ReLU()
        )
        self.conv4 = Sequential(
            Conv3d(in_channels=num_labels, out_channels=num_labels, kernel_size=(3, 7, 7), padding=(1, 3, 3)),
            Tanh()
        )

        self.upsampling = Upsample(scale_factor=(1, self.downsampling_factor, self.downsampling_factor),
                                   mode='trilinear', align_corners=True)

    def forward(self, inputs, **kwargs):
        # local part
        node = self.scnet_local(inputs)

        return node


class LocNet(nn.Module):
    def __init__(self,
                 num_labels,
                 num_filters_base=64,
                 spatial_downsample=4,
                 residual='conv',
                 channels=1):
        super(LocNet, self).__init__()
        self.conv1 = nn.Conv2d()


if __name__ == '__main__':
    net = SpatialConfigurationNet(num_labels=20, residual='pool')
    writer = SummaryWriter()
    x = torch.ones(1, 1, 12, 64, 64)
    print(x.size())
    writer.add_graph(net, (x,))
    h = net.forward(x)
    print(h.size())
    writer.close()