import argparse
from src.normalize import normalize_images, ImgLoader, save_images
from pathlib import Path
from tqdm import tqdm
import shutil


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inpath', required=True, help='The path to the input dir.')
    parser.add_argument('--outpath', required=True, help='The path to the output dir.')
    parser.add_argument('--bsize', required=True, help='The number of images to use in a batch.')

    args = parser.parse_args()

    print(f"\nNormalizing images in {args.inpath} and saving to {args.outpath}")

    path = Path(args.outpath)
    if path.exists():
        delete = input('Folder already exists. Delete? [y/n]:')
        if bool(delete):
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)

    loader = ImgLoader(args.inpath, batch_size=int(args.bsize))
    loop = tqdm(enumerate(loader), total=len(loader))

    for batch, (images, files) in loop:
        normalized_images, files = normalize_images(images, files)
        save_images(normalized_images, out_path=args.outpath, file_names=files)


if __name__ == '__main__':
    main()