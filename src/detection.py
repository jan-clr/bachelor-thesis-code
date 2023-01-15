import numpy as np
import cv2
import torch
from skimage import measure


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def detect_droplets(image):
    image[image != 2] = 0
    droplets = measure.label(image)
    nr_droplets = np.unique(droplets) - 1
    circles = []
    for i in range(1, nr_droplets + 2):
        pixel_locs = np.where(droplets == i)
        center = np.mean(pixel_locs, dim=0)
        distances = []
        for loc in pixel_locs:
            distances.append(np.linalg.norm(loc - center))
        radius = np.mean(np.array(distances))
        circles.append({'center': center, 'radius': radius})

    return circles
