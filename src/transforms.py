import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, InterpolationMode


def transform(image, target):
	"""

	:param image:
	:param target:
	:return:
	"""
	size = (224, 224)

	# resize the image and the mask
	resize_img = Resize(size=size)
	resize_target = Resize(size=size, interpolation=InterpolationMode.NEAREST)
	image = resize_img(image)
	target = resize_target(target)

	# convert to Tensors
	image = TF.to_tensor(image)
	target = torch.as_tensor(np.array(target), dtype=torch.long)

	return image, target
