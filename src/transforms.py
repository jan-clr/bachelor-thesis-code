import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop


def transforms_train(image, target):
	"""

	:param image:
	:param target:
	:return:
	"""
	size = (224, 224)

	# convert to Tensors
	image = TF.to_tensor(image)
	target = torch.as_tensor(np.array(target), dtype=torch.long)

	# Random crop
	i, j, h, w = RandomCrop.get_params(
		image, output_size=size)
	image = TF.crop(image, i, j, h, w)
	target = TF.crop(target, i, j, h, w)

	return image, target


def transforms_val(image, target):
	# convert to Tensors
	image = TF.to_tensor(image)
	target = torch.as_tensor(np.array(target), dtype=torch.long)

	return image, target