import numpy as np
import cv2
import torch
from skimage import measure
from dataclasses import dataclass
from typing import List
import warnings


def load_image(path, greyscale=True):
    if greyscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if img is None:
        print(f"Image at {path} could not be read.")
    return img


def detect_droplets(image, lower_percentile=80, upper_percentile=95, label_id=1):
    image[image != label_id] = 0
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
        used_distances = [d for d in distances if
                          np.percentile(distances, lower_percentile) <= d <= np.percentile(distances, upper_percentile)]
        if used_distances:
            radius = np.mean(np.array(used_distances))
            circles.append(Droplet(center=center, radius=radius))

    return circles


def detect_streaks(image, label_id=3):
    image[image != label_id] = 0
    droplets = measure.label(image)
    labels = np.unique(droplets)[1:]
    nr_droplets = len(labels)
    rects = []
    for i in labels:
        pixel_locs = np.argwhere(droplets == i)
        rect = cv2.minAreaRect(pixel_locs)
        rects.append(rect)

    return rects


@dataclass
class Droplet:
    center: (int, int)
    radius: float | int


def draw_droplets(img, droplets: List[Droplet]):
    for droplet in droplets:
        cv2.circle(img, droplet.center[::-1], np.round(droplet.radius).astype(int), (0, 255, 0))

    return img


def draw_streaks(img, streaks):
    return img


def main():
    detect_streaks()


if __name__ == '__main__':
    main()