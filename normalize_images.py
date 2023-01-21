import argparse
from src.normalize import normalize_images_batched


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath', required=True, help='The path to the input dir.')
    parser.add_argument('--outpath', required=True, help='The path to the output dir.')
    parser.add_argument('--bsize', required=True, help='The number of images to use in a batch.')

    args = parser.parse_args()

    print(f"\nNormalizing images in {args.inpath} and saving to {args.outpath}")

    normalize_images_batched(args.inpath, args.outpath, args.bsize)


if __name__ == '__main__':
    main()