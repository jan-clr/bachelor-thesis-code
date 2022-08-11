import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize


def transform(image, target):
	"""

	:param image:
	:param target:
	:return:
	"""

	# resize the image and the mask
	resize = Resize(size=(224, 224))
	image = resize(image)
	target = resize(target)

	# convert to Tensors
	image = TF.to_tensor(image)
	target = torch.as_tensor(np.array(target), dtype=torch.long)

	return image, target
