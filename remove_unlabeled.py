import os
from pathlib import Path
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeldir', type=str, default='data/vapourbase_DS2/gtFine/train/50Hz-1_50000el_1cmAbstand_36dBm_65mu')
    parser.add_argument('--imdir', type=str, default='data/vapourbase_DS2/leftImg8bit/train/50Hz-1_50000el_1cmAbstand_36dBm_65mu')

    args = parser.parse_args()

    images = glob.glob(f"{args.imdir}/*.png", recursive=True)
    nr_unlabeled = 0
    for img in images:
        label = img.replace(args.imdir, args.labeldir).replace('leftImg8bit', 'gtFine_labelIds')
        if not os.path.exists(label):
            print(f"Removing {img}")
            nr_unlabeled += 1
            os.remove(img)

    print(f"Removed {nr_unlabeled} unlabeled images from {len(images)} images. {len(images) - nr_unlabeled} images left.")


if __name__ == '__main__':
    main()