import torch
from torch import nn
from tensorboardX import SummaryWriter

# limit from archs import *
__all__ = ["UNet", "NestedUNet", "AttentionUNet", "R2UNet"]


class VGGBlock(nn.Module):
	def __init__(self, in_channels, middle_channels, out_channels):
		super(VGGBlock, self).__init__()

		self.relu = nn.ReLU(inplace = True)

		## nn.Conv2d
		## torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
		self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)

		## nn.BatchNorm2d
		## torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		## BatchNorm层的作用是在深度学习网络训练过程使得每一层神经网络的输入保持相同分布。
		self.bn1 = nn.BatchNorm2d(middle_channels)
		self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		return out

class up_conv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(up_conv, self).__init__()

		self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
		self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):
		out = self.up(x)
		out = self.conv(out)
		out = self.bn(out)
		out = self.relu(out)

		return out


class UNet(nn.Module):
	def __init__(self, num_classes, in_channels=3, *args, **kwargs):
		super().__init__()

		n1 = 64
		filters = [n1, 2*n1, 4*n1, 8*n1, 16*n1]

		self.maxpool = nn.MaxPool2d(2, 2)

		self.conv1 = VGGBlock(in_channels, filters[0], filters[0])
		self.conv2 = VGGBlock(filters[0], filters[1], filters[1])
		self.conv3 = VGGBlock(filters[1], filters[2], filters[2])
		self.conv4 = VGGBlock(filters[2], filters[3], filters[3])
		self.conv5 = VGGBlock(filters[3], filters[4], filters[4])

		self.up5 = up_conv(filters[4], filters[3])
		self.up5_conv = VGGBlock(filters[4], filters[3], filters[3])
		self.up4 = up_conv(filters[3], filters[2])
		self.up4_conv = VGGBlock(filters[3], filters[2], filters[2])
		self.up3 = up_conv(filters[2], filters[1])
		self.up3_conv = VGGBlock(filters[2], filters[1], filters[1])
		self.up2 = up_conv(filters[1], filters[0])
		self.up2_conv = VGGBlock(filters[1], filters[0], filters[0])

		self.convFinal = nn.Conv2d(filters[0], num_classes, kernel_size=1, padding=0)


	def forward(self, input):
		e1 = self.conv1(input)

		e2 = self.maxpool(e1)
		e2 = self.conv2(e2)

		e3 = self.maxpool(e2)
		e3 = self.conv3(e3)

		e4 = self.maxpool(e3)
		e4 = self.conv4(e4)

		e5 = self.maxpool(e4)
		e5 = self.conv5(e5)

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

		output = self.convFinal(d2)

		return output

class NestedUNet(nn.Module):
	def __init__(self, num_classes=1, in_channels=3, deep_supervision=False, *args, **kwargs):
		super(NestedUNet, self).__init__()

		n1 = 64
		filters = [n1, 2*n1, 4*n1, 8*n1, 16*n1]

		self.deep_supervision = deep_supervision
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

		self.conv0_0 = VGGBlock(in_channels, filters[0], filters[0])
		self.conv1_0 = VGGBlock(filters[0], filters[1], filters[1])
		self.conv2_0 = VGGBlock(filters[1], filters[2], filters[2])
		self.conv3_0 = VGGBlock(filters[2], filters[3], filters[3])
		self.conv4_0 = VGGBlock(filters[3], filters[4], filters[4])


		self.conv0_1 = VGGBlock(filters[0] + filters[1], filters[0], filters[0])
		self.conv1_1 = VGGBlock(filters[1] + filters[2], filters[1], filters[1])
		self.conv2_1 = VGGBlock(filters[2] + filters[3], filters[2], filters[2])
		self.conv3_1 = VGGBlock(filters[3] + filters[4], filters[3], filters[3])

		self.conv0_2 = VGGBlock(filters[0]*2 + filters[1], filters[0], filters[0])
		self.conv1_2 = VGGBlock(filters[1]*2 + filters[2], filters[1], filters[1])
		self.conv2_2 = VGGBlock(filters[2]*2 + filters[3], filters[2], filters[2])

		self.conv0_3 = VGGBlock(filters[0]*3 + filters[1], filters[0], filters[0])
		self.conv1_3 = VGGBlock(filters[1]*3 + filters[2], filters[1], filters[1])

		self.conv0_4 = VGGBlock(filters[0]*4 + filters[1], filters[0], filters[0])

		# self.convFinal = nn.Conv2d(filters[0], out_channels, kernel_size=1, padding=0)

		if self.deep_supervision:
			self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
			self.final2 = nn.Conv2d(filters[0], num_classes, kernel_szie=1)
			self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
			self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
		else:
			self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)


	def forward(self, input):
		x0_0 = self.conv0_0(input)
		x1_0 = self.conv1_0(self.maxpool(x0_0))
		x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

		x2_0 = self.conv2_0(self.maxpool(x1_0))
		x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

		x3_0 = self.conv3_0(self.maxpool(x2_0))
		x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

		x4_0 = self.conv4_0(self.maxpool(x3_0))
		x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
		x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
		x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
		x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
		
	
		if self.deep_supervision:
			output1 = self.final1(x0_1)
			output2 = self.final2(x0_2)
			output3 = self.final3(x0_3)
			output4 = self.final4(x0_4)
			return [output1, output2, output3, output4]
		else:
			output = self.final(x0_4)
			return output

class AttentionBlock(nn.Module):
	def __init__(self, F_g, F_l, F_int):
		super().__init__()

		self.w_g = nn.Sequential(
				nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
				nn.BatchNorm2d(F_int)
			)
		self.w_x = nn.Sequential(
				nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
				nn.BatchNorm2d(F_int)
			)
		self.psi = nn.Sequential(
				nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
				nn.BatchNorm2d(1),
				nn.Sigmoid()
			)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, g, x):
		g1 = self.w_g(g)
		x1 = self.w_x(x)
		psi = self.relu(g1 + x1)
		psi = self.psi(psi)
		out = x * psi
		return out


class AttentionUNet(nn.Module):
	'''
	Paper: https://arxiv.org/abs/1804.03999
	'''
	def __init__(self, classes=1, in_channels=3, *args, **kwargs):
		super().__init__()

		n1 = 64
		filters = [n1, 2*n1, 4*n1, 8*n1, 16*n1]

		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv1 = VGGBlock(in_channels, filters[0], filters[0])
		self.conv2 = VGGBlock(filters[0], filters[1], filters[1])
		self.conv3 = VGGBlock(filters[1], filters[2], filters[2])
		self.conv4 = VGGBlock(filters[2], filters[3], filters[3])
		self.conv5 = VGGBlock(filters[3], filters[4], filters[4])

		self.up5 = up_conv(filters[4], filters[3])
		self.Att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
		self.up5_conv = VGGBlock(filters[4], filters[3], filters[3])

		self.up4 = up_conv(filters[3], filters[2])
		self.Att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
		self.up4_conv = VGGBlock(filters[3], filters[2], filters[2])
		
		self.up3 = up_conv(filters[2], filters[1])
		self.Att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
		self.up3_conv = VGGBlock(filters[2], filters[1], filters[1])
		
		self.up2 = up_conv(filters[1], filters[0])
		self.Att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=int(filters[0]/2))
		self.up2_conv = VGGBlock(filters[1], filters[0], filters[0])

		self.convFinal = nn.Conv2d(filters[0], classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		e1 = self.conv1(x)

		e2 = self.maxpool(e1)
		e2 = self.conv2(e2)

		e3 = self.maxpool(e2)
		e3 = self.conv3(e3)

		e4 = self.maxpool(e3)
		e4 = self.conv4(e4)

		e5 = self.maxpool(e4)
		e5 = self.conv5(e5)

		d5 = self.up5(e5)	
		x4 = self.Att5(g=d5, x=e4)
		d5 = torch.cat((x4, d5), dim=1)
		d5 = self.up5_conv(d5)

		d4 = self.up4(d5)
		x3 = self.Att4(g=d4, x=e3)
		d4 = torch.cat((x3, d4), dim=1)
		d4 = self.up4_conv(d4)

		d3 = self.up3(d4)
		x2 = self.Att3(g=d3, x=e2)
		d3 = torch.cat((x2, d3), dim=1)
		d3 = self.up3_conv(d3)

		d2 = self.up2(d3)
		x1 = self.Att2(g=d2, x=e1)
		d2 = torch.cat((x1, d2), dim=1)
		d2 = self.up2_conv(d2)

		output = self.convFinal(d2)

		return output

class RecurrentBlock(nn.Module):
	def __init__(self, in_channels, t=2):
		super().__init__()

		self.t = t
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		for i in range(self.t):
			if i == 0:
				out = self.conv(x)
			out = self.conv(x + out)
		return out

class RRCNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, t=2):
		super().__init__()

		self.rb1 = RecurrentBlock(in_channels=out_channels, t=t)
		self.rb2 = RecurrentBlock(in_channels=out_channels, t=t)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		x1 = self.conv(x)
		x2 = self.rb1(x1)
		x3 = self.rb2(x2)

		out = x1 + x3
		return out

class R2UNet(nn.Module):
	'''
	Paper: https://arxiv.org/abs/1802.06955
	'''
	def __init__(self, classes=1, in_channels=3, *args, **kwargs):
		super().__init__()

		n1 = 64
		t = 2
		filters = [n1, 2*n1, 4*n1, 8*n1, 16*n1]

		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.RRCNN1 = RRCNNBlock(in_channels, filters[0], t=t)
		self.RRCNN2 = RRCNNBlock(filters[0], filters[1], t=t)
		self.RRCNN3 = RRCNNBlock(filters[1], filters[2], t=t)
		self.RRCNN4 = RRCNNBlock(filters[2], filters[3], t=t)
		self.RRCNN5 = RRCNNBlock(filters[3], filters[4], t=t)

		self.up5 = up_conv(filters[4], filters[3])
		self.up5_conv = RRCNNBlock(filters[4], filters[3], t)

		self.up4 = up_conv(filters[3], filters[2])
		self.up4_conv = RRCNNBlock(filters[3], filters[2], t)

		self.up3 = up_conv(filters[2], filters[1])
		self.up3_conv = RRCNNBlock(filters[2], filters[1], t)

		self.up2 = up_conv(filters[1], filters[0])
		self.up2_conv = RRCNNBlock(filters[1], filters[0], t)


		self.convFinal = nn.Conv2d(filters[0], classes, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		e1 = self.RRCNN1(x)

		e2 = self.maxpool(e1)
		e2 = self.RRCNN2(e2)

		e3 = self.maxpool(e2)
		e3 = self.RRCNN3(e3)

		e4 = self.maxpool(e3)
		e4 = self.RRCNN4(e4)

		e5 = self.maxpool(e4)
		e5 = self.RRCNN5(e5)

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

		output = self.convFinal(d2)

		return output

class ConvFour(nn.Module):
	def __init__(self, in_channels=1):
		super().__init__()

		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
		self.bn2 = nn.BatchNorm2d(in_channels)
		self.conv3 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
		self.bn3 = nn.BatchNorm2d(in_channels)
		self.conv4 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
		self.bn4 = nn.BatchNorm2d(in_channels)

	def forward(self, x):
		out = self.conv1(x)
		# out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		# out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		# out = self.bn3(out)
		out = self.relu(out)

		out = self.conv4(out)
		# out = self.bn4(out)
		out = self.relu(out)

		return out