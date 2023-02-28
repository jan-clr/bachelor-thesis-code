import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import VapourData
from src.detection import detect_droplets_hough, draw_droplets
from src.transforms import transforms_hough
from src.utils import DropletAccuracyHough


def main():
    val_data = VapourData('data/vapourbase_v3', mode='test', transforms=transforms_hough, split=False)
    loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
    # loader = ImgLoader('data/vapourbase_v3/leftImg8bit/test/0', batch_size=1, grayscale=True)
    accuracy = DropletAccuracyHough()
    loop = tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, targets) in loop:
        accuracy.update(imgs, targets)
        if i % 5 != 0:
            continue
        for j, img in enumerate(imgs):
            img = np.ascontiguousarray(img.numpy().transpose(1, 2, 0), dtype=np.uint8)
            droplets = detect_droplets_hough(img)
            img = draw_droplets(img, droplets)
            cv2.imshow('detected circles', img)
            cv2.waitKey()
            cv2.destroyAllWindows()

    metrics = accuracy.compute()
    print(metrics)


if __name__ == '__main__':
    main()