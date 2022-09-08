import torch
import torch.nn as nn


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

	def forward(self, x):
		return self.conv(x)


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
	x = torch.randn((1, 3, 224, 224))
	model = CS_UNET(in_ch=3, out_ch=3)
	preds = model(x)
	print(x.shape, preds.shape)
	assert x.shape == preds.shape


def main():
	test()


if __name__ == '__main__':
	main()
