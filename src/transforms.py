import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
import albumentations as A
from albumentations.pytorch import ToTensorV2


def transforms_train(image, mask):
	"""

	:param image:
	:param mask:
	:return:
	"""
	image = np.array(image)
	mask = np.array(mask)

	transform = A.Compose([
		A.RandomCrop(224, 224),
		A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
		A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
		A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	])

	augmented = transform(image=image, mask=mask)
	image = augmented["image"]
	mask = augmented["mask"].long()

	return image, mask


def transforms_val(image, mask):
	image = np.array(image)
	mask = np.array(mask)

	transform = A.Compose(
		[A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
	)
	augmented = transform(image=image, mask=mask)
	image = augmented["image"]
	mask = augmented["mask"].long()

	return image, mask


def transforms_train_torch(image, mask):
	"""

	:param image:
	:param mask:
	:return:
	"""
	size = (224, 224)

	# convert to Tensors
	image = TF.to_tensor(image)
	mask = torch.as_tensor(np.array(mask), dtype=torch.long)

	# Random crop
	i, j, h, w = RandomCrop.get_params(
		image, output_size=size)
	image = TF.crop(image, i, j, h, w)
	mask = TF.crop(mask, i, j, h, w)

	return image, mask


def transforms_val_torch(image, mask):
	# convert to Tensors
	image = TF.to_tensor(image)
	mask = torch.as_tensor(np.array(mask), dtype=torch.long)

	return image, mask
