import numpy as np
import cv2
from skimage import measure
from dataclasses import dataclass
import alphashape
from shapely import Point
from shapely import Polygon
from itertools import product
from scipy.ndimage import gaussian_filter



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


def hough_transform(image, min_radius=3, max_radius=20):
    blurred = gaussian_filter(image, sigma=1.5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, 1.5, 20, param1=30, param2=0.8, minRadius=min_radius,
                               maxRadius=max_radius)
    if circles is None:
        return []

    return circles[0, :]
    #circles = np.round(circles[0, :]).astype("int")


def detect_droplets_hough(image, min_radius=3, max_radius=20):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = hough_transform(image, min_radius, max_radius)
    droplets = []
    for (x, y, r) in circles:
        contrast = contrast_in_circle(image, np.round([y, x]).astype(int), r)
        print(contrast)
        if contrast < 0.3:
            continue
        droplets.append(Droplet(center=np.round([y, x]).astype(int), radius=r))

    return droplets


def coords_in_circle(center, radius):
    min_y, min_x, max_y, max_x = [int(x) for x in [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]]
    coord_array = [(y, x) for y, x in product(range(min_y, max_y + 1), range(min_x, max_x + 1)) if np.linalg.norm(np.array([y, x]) - center) <= radius]
    return coord_array


def contrast_in_circle(image, center, radius):
    circle_coords = coords_in_circle(center, radius)
    circle_pixels = [image[y, x] for y, x in circle_coords if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]]
    return michelson_contrast(circle_pixels)


def michelson_contrast(values):
    return (np.max(values) - float(np.min(values))) / (np.max(values) + float(np.min(values)))

def main():
    pass


if __name__ == '__main__':
    main()