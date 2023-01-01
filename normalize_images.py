import argparse
from src.normalize import normalize_images, ImgLoader, save_images
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath', required=True, help='The path to the input dir.')
    parser.add_argument('--outpath', required=True, help='The path to the output dir.')
    parser.add_argument('--bsize', required=True, help='The number of images to use in a batch.')

    args = parser.parse_args()

    loader = ImgLoader(args.inpath, batch_size=int(args.bsize))
    loop = tqdm(enumerate(loader), total=len(loader))

    print(f"\nNormalizing images in {args.inpath} and saving to {args.outpath}")

    path = Path(args.outpath)
    path.mkdir(parents=True, exist_ok=False)

    for batch, (images, files) in loop:
        normalized_images = normalize_images(images)
        save_images(normalized_images, out_path=args.outpath, file_names=files)


if __name__ == '__main__':
    main()