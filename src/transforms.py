import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TransformTwice:
    """
    Slight alteration to mean_teachers TransformTwice to be Compatible with Albumentations
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(**inp)
        out2 = self.transform(**inp)
        return out1, out2


def transforms_train_mt(image, mask):
    """

    :param image:
    :param mask:
    :return:
    """
    image = np.array(image)
    if mask is not None:
        mask = np.array(mask)

    transform_same = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
    ])

    transform_student = A.Compose([
        A.GaussNoise(var_limit=0.15),
        A.ColorJitter(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transform_teacher = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    images = None

    if mask is not None:
        augmented_same = transform_same(image=image, mask=mask)
        image_student = transform_student(image=augmented_same['image'])
        image_teacher = transform_teacher(image=augmented_same['image'])
        images = (image_student['image'], image_teacher['image'])
        # Mask alone cannot be transformed
        mask = ToTensorV2()(image=np.zeros_like(image), mask=augmented_same["mask"])['mask'].long()
    else:
        augmented_same = transform_same(image=image)
        image_student = transform_student(image=augmented_same['image'])
        image_teacher = transform_teacher(image=augmented_same['image'])
        images = (image_student['image'], image_teacher['image'])

    return images, mask


def transforms_train_mt_basic(image, mask):
    """
        Performs only basic transforms
        :param image:
        :param mask:
        :return:
    """
    image = np.array(image)
    if mask is not None:
        mask = np.array(mask)

    transform_same = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    if mask is not None:
        augmented_same = transform_same(image=image, mask=mask)
        image, mask = augmented_same['image'], augmented_same['mask'].long()
    else:
        image = transform_same(image=image)['image']

    return image, mask


def transforms_generator(image, mask):
    image, mask = np.array(image), np.array(mask)
    tt = ToTensorV2()
    transformed = tt(image=image, mask=mask)
    return transformed['image'], transformed['mask']


def apply_transforms(transforms, image, mask):
    """

        :param image:
        :param mask:
        :return:
        """
    image = np.array(image)
    if mask is not None:
        mask = np.array(mask)

    if mask is not None:
        augmented = transforms(image=image, mask=mask)
        mask = augmented["mask"].long()
    else:
        augmented = transforms(image=image)

    image = augmented["image"]

    return image, mask


def transforms_train_cs(image, mask):
    transform = A.Compose([
        A.RandomCrop(224, 224),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        # A.ColorJitter(),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # A.GaussNoise(var_limit=0.15),
        # Norm at end https://stats.stackexchange.com/questions/337237/order-of-normalization-augmentation-for-image-classification
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return apply_transforms(transform, image, mask)


def transforms_train_vap(image, mask):
    transform = A.Compose([
        A.RandomCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussNoise(var_limit=0.15),
        ToTensorV2(),
    ])
    return apply_transforms(transform, image, mask)


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


def transforms_hough(image, mask):
    image = np.array(image)
    mask = np.array(mask)

    transform = ToTensorV2()
    augmented = transform(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"].long()

    return image, mask


def transform_eval(image):
    image = np.array(image)

    transform = A.Compose(
        [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    augmented = transform(image=image)
    image = augmented["image"]

    return image


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


def gauss_noise_tensor(img):
    """
    Add gaussian noise to a tensor
    https://github.com/pytorch/vision/issues/6192#issuecomment-1164176231
    :param img: 
    :return: 
    """
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 0.15

    out = img + sigma * torch.randn_like(img)
    out = torch.clamp(out, 0, 1)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out