from src.detection import load_image
from glob import glob
import numpy as np


def main():
    root = 'data/job_4-2023_02_19_16_02_22-cityscapes 1.0/gtFine/default'
    files = glob(f"{root}/**/*labelIds*", recursive=True)
    counts = {}
    total = 0
    for file in files:
        img = load_image(file)
        for label in np.unique(img):
            if label in counts.keys():
                counts[label] += np.sum(img == label)
            else:
                counts[label] = np.sum(img == label)
        total += img.shape[0] * img.shape[1]

    for label, count in counts.items():
        print(f"id: {label:>2} weight: {count / total}")


if __name__ == '__main__':
    main()