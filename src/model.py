import torch
import torch.nn as nn
import timm
import itertools


class DoubleConv(nn.Module):
	"""
    Performs two same convolutions back to back, accepting an input with in_ch channels and outputting out_ch channels
    """

	def __init__(self, in_ch: int, out_ch: int):
		"""
        :param in_ch: nr of input channels
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

	def __init__(self, in_ch=3, out_ch=2, encoder_name='resnet34', freeze_encoder=False, dropout_p=None):
		super(UnetResEncoder, self).__init__()

		if dropout_p is not None:
			self.dropout = nn.Dropout(p=dropout_p)
		else:
			self.dropout = None

		self.encoder = timm.create_model(encoder_name, pretrained=True, features_only=True, in_chans=in_ch)
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


def test():
	x = torch.randn((4, 3, 512, 256))
	model = UnetResEncoder()
	model.eval()
	out = model(x)
	for o in out:
		print(o.shape)


def main():
	test()


if __name__ == '__main__':
	main()
