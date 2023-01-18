import cv2
import argparse
import numpy as np

from src.detection import detect_droplets, load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', metavar='FP', help='The path of the image to detect droplets on.')

    args = parser.parse_args()

    img = load_image(path=args.filepath)

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
    cv2.normalize(heatmap, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    print(np.unique(heatmap))

    droplets = detect_droplets(img)
    print(droplets)
    for circ in droplets:
        cv2.circle(heatmap, circ['center'][::-1], circ['radius'], (0, 255, 0))

    cv2.imshow('Image', heatmap)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()