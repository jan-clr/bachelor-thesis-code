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

