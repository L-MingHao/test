import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, groups):
        super(SingleConv, self).__init__()

        self.single_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, groups=groups),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)

class UpSampling(nn.Module):
    def __init__(self, scale_factor=(1, 2, 2)):
        super(UpSampling, self).__init__()

        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        return x



class SpatialConfiguration(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpatialConfiguration, self).__init__()
        self.AveragePooling = nn.AvgPool3d(3, stride=2, padding=1)
        self.DepthwiseConv = SingleConv(5, 5, kernel_size=3, stride=1, padding=1, groups=5)
        self.PointwiseConv = SingleConv(5, 5, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        output = self.AveragePooling(x)
        output = self.DepthwiseConv(output)
        output = self.PointwiseConv(output)

        return output



class Appearance(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Appearance, self).__init__()
        self.CovOnly = nn.Sequential(
            SingleConv(in_ch, 32, kernel_size=3, stride=1, padding=1, groups=1),
            SingleConv(32, 5, kernel_size=3, stride=1, padding=1, groups=1)
        )

    def forward(self, x):
        output = self.CovOnly(x)

        return output


class BaseNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BaseNet, self).__init__()
        self.appearance = Appearance(in_ch, out_ch)
        self.spatialconfiguration = SpatialConfiguration(in_ch,out_ch)
        self.upsampling = UpSampling(scale_factor=(2, 2, 2))

        # init
        self.initialize()

    @staticmethod
    def init_conv_IN(modules):
        for m in modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.appearance.modules)
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_IN(self.spatialconfiguration.modules)


    def forward(self, x):
        Appearance_output = self.appearance(x)
        SpatialConfiguration_output = self.spatialconfiguration(Appearance_output)
        UpSampling_output = self.upsampling(SpatialConfiguration_output)
        Combination = Appearance_output * UpSampling_output

        return Combination

class Model(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Model, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.net = BaseNet(in_ch, out_ch)
        self.conv_out = nn.Conv3d(5, out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        output = self.net(x) #input (2,1,D,H,W)
        output = self.conv_out(output)
        return output