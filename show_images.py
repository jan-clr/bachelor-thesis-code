from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(prog='ImageInspector', description='Inspect images on disk as masks or actual images.')

    parser.add_argument('paths', metavar='path', type=str, help='The paths of the images to read.', nargs='+')

    args = parser.parse_args()

    print(args)
    for path in args.paths:
        img = Image.open(path).convert('L')
        ar = np.array(img)
        print(ar.dtype)
        plt.imshow(ar)
    plt.show()


if __name__ == '__main__':
    main()