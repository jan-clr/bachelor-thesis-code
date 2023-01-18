import numpy as np
import cv2
import torch
from skimage import measure


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def detect_droplets(image, lower_percentile=80, upper_percentile=95):
    image[image != 2] = 0
    droplets = measure.label(image)
    nr_droplets = len(np.unique(droplets)) - 1
    circles = []
    for i in range(1, nr_droplets + 1):
        pixel_locs = np.argwhere(droplets == i)
        center = np.round(np.mean(pixel_locs, axis=0)).astype(int)
        distances = []
        for loc in pixel_locs:
            distances.append(np.linalg.norm(loc - center))
        distances.sort()
        used_distances = [d for d in distances if np.percentile(distances, lower_percentile) <= d <= np.percentile(distances, upper_percentile)]
        radius = np.round(np.mean(np.array(used_distances))).astype(int)
        circles.append({'center': center, 'radius': radius})

    return circles
