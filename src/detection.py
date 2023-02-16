import numpy as np
import cv2
import torch
from skimage import measure
from dataclasses import dataclass
import alphashape
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from shapely import Point
from shapely import Polygon
from itertools import product


def load_image(path, greyscale=True):
    if greyscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if img is None:
        print(f"Image at {path} could not be read.")
    return img


def detect_droplets(image, lower_percentile=80, upper_percentile=95, label_id=1, exlude_small=True):
    image_cpy = np.copy(image)
    image_cpy[image_cpy != label_id] = 0
    droplets = measure.label(image_cpy)
    nr_droplets = len(np.unique(droplets)) - 1
    circles = []
    for i in range(1, nr_droplets + 1):
        pixel_locs = np.argwhere(droplets == i)
        # if droplet doesn't have center pixels inside it is unlikely to be a sharp droplet
        if not droplet_has_inside(pixel_locs, image):
            continue
        center = np.round(np.mean(pixel_locs, axis=0)).astype(int)
        distances = []
        for loc in pixel_locs:
            distances.append(np.linalg.norm(loc - center))
        distances.sort()
        used_distances = [d for d in distances if
                          np.percentile(distances, lower_percentile) <= d <= np.percentile(distances, upper_percentile)]
        if len(used_distances) > 5:
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


def draw_droplets(img, droplets: [Droplet]):
    for droplet in droplets:
        cv2.circle(img, droplet.center[::-1], np.round(droplet.radius).astype(int), (0, 255, 0))

    return img


def droplet_has_inside(droplet_coords, image, check_for=2):
    if len(droplet_coords) < 5:
        return False
    # compute alphashape of the droplet label
    try:
        ashape = alphashape.alphashape(droplet_coords, alpha=1)
        if ashape.geom_type != 'Polygon':
            return False
        min_x, min_y, max_x, max_y = [int(x) for x in ashape.bounds]
        # Polygon is out of bounds
        if min_x < 0 or min_y < 0 or max_x >= image.shape[0] or max_y >= image.shape[1]:
            print("Polygon out of Bounds")
            return False
        inner_cords = np.array(get_points_in_poly(polygon=ashape))
        if len(inner_cords) <= 0:
            return False
        inside_pixels = np.sum(image[inner_cords.T[0], inner_cords.T[1]] == check_for)
        return inside_pixels > 0
    except Exception as err:
        print(err)
        return False


def get_points_in_poly(polygon: Polygon):
    min_x, min_y, max_x, max_y = [int(x) for x in polygon.bounds]
    coord_array = [[x, y] for x, y in product(range(min_x, max_x + 1), range(min_y, max_y + 1)) if polygon.contains(Point([x, y]))]
    return coord_array


def draw_streaks(img, streaks):
    return img


def main():
    detect_streaks()


if __name__ == '__main__':
    main()