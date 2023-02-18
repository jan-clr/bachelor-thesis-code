import argparse
import os
from src.normalize import normalize_images_batched


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath', required=True, help='The path to the input dir.')
    parser.add_argument('--outpath', help='The path to the output dir.')
    parser.add_argument('--bsize', type=int, help='The number of images to use in a batch.', default=20)

    args = parser.parse_args()

    outpath = args.outpath or os.path.join(args.inpath, 'normalized')

    print(f"\nNormalizing images in {args.inpath} and saving to {args.outpath}")

    normalize_images_batched(args.inpath, outpath, args.bsize)


if __name__ == '__main__':
    main()