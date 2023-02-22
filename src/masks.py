import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import numpy as np


class CowMaskGenerator(Dataset):
    def __init__(self, crop_size=(256, 256), method='cut', sigmas=None):
        """
        Generates Cow Masks only
        https://arxiv.org/abs/2210.10426
        """
        self.crop_size = crop_size
        self.method = method
        if sigmas is None:
            self.sigmas = (13, 15, 17, 19, 21, 23, 25)
        else:
            self.sigmas = sigmas

    def __getitem__(self, idx):
        cow_size = self.crop_size
        sigma = np.random.choice(self.sigmas)
        mask = generate_cow_mask(size=cow_size, sigma=sigma, method=self.method)
        mask = np.expand_dims(mask, axis=0)

        return mask

    def __len__(self):
        return 2000000000


def generate_cow_mask(size, sigma, method):
    """
    Generates a cow mask
    https://arxiv.org/abs/2210.10426
    :param size:
    :param sigma:
    :param method:
    :return:
    """
    cow_mask = np.random.uniform(low=0.0, high=1.0, size=size)
    cow_mask_gauss = gaussian_filter(cow_mask, sigma=sigma)

    mean = np.mean(cow_mask_gauss)
    std = np.std(cow_mask_gauss)
    # thresh = mean + perturbation*std
    if method == "mix":
        cow_mask_final = (cow_mask_gauss < mean).astype(np.int32)
    elif method == "cut":
        offset = np.random.uniform(low=0.5, high=1.0, size=())
        cow_mask_final = (cow_mask_gauss < mean + offset * std).astype(np.int32)
    else:
        raise NotImplementedError

    return cow_mask_final


class CutMixMaskGenerator(Dataset):
    def __init__(self, crop_size=(256, 256)):
        """
        Generates CutMix Masks only
        https://arxiv.org/abs/1905.04899
        """
        self.crop_size = crop_size

    def __getitem__(self, idx):
        mask = generate_cutmix_mask(size=self.crop_size)
        mask = np.expand_dims(mask, axis=0)

        return mask

    def __len__(self):
        return 2000000000


def generate_cutmix_mask(size):
    """
    Generates a cutmix mask
    https://arxiv.org/abs/1905.04899
    :param size: The mask size
    :return:
    """
    mask = np.ones(size)
    lam = np.random.beta(1.0, 1.0)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, 0.5)
    mask[bbx1:bbx2, bby1:bby2] = 0.0

    return mask


def rand_bbox(size, lam):
    """
    Generates a random bounding box
    https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
    :param size: The mask size
    :param lam: Value to determine cut ration. high lambda = small cut
    :return:
    """
    W = size[1]
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def main():
    mask_gen = CutMixMaskGenerator((256, 512))
    mask = mask_gen[0]
    print(mask.shape)
    mask = mask.squeeze()
    print(mask.shape)
    cv2.imshow("mask", (mask * 255).astype(np.uint8))
    print(np.sum(mask == 0))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()