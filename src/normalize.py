import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import shutil
from pathlib import Path
from src.utils import save_images, ImgLoader
from tqdm import tqdm

CONTRAST_THRESHOLD = 0.6


def normalize_data(path) -> str:
    """
    Normalizes all image files in a provided directory and saves the normalized copies to path + '_normalized'

    :param path: The path of the directory with the files to be normalized.
    :return: The path to the normalized images
    """
    if not os.path.isdir(path):
        print('Path is not a directory')
        quit()

    images = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        parts = file.split('.')
        name, ext = parts[0], parts[1]
        if os.path.isfile(file_path):
            images.append({'path': file_path, 'ext': ext, 'name': name})

    print(f"Found {len(images)} images.")

    norm_dir = f"{path}_normalized"
    if not os.path.isdir(norm_dir):
        os.mkdir(norm_dir)

    nr_norm = 0
    print("")
    for img in images:
        cv_img = cv2.imread(img['path'])
        if cv_img is not None:
            img_norm = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imshow('Test', img_norm)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(f"{norm_dir}/{img['name']}_normalized.{img['ext']}", img_norm)
            nr_norm += 1
            print(f"\rNormalized {nr_norm}/{len(images)}", end='')

    print(f"\nNormalized images saved to '{norm_dir}'")

    return norm_dir


def normalize_images(images, files, remove_low_contrast=True, overwrite_artifacts=True):
    images = images.astype('int32')
    mean_img = np.mean(images, axis=0)
    mean_img = mean_img.astype('uint8')
    mean_val = np.mean(images)

    images = (images - mean_img + mean_val)
    min_val = np.min(images)
    max_val = np.max(images)
    images = ((images - min_val) / (max_val - min_val) * 255.0).astype('uint8')

    if overwrite_artifacts:
        mean_blurred = gaussian_filter(mean_img, sigma=3)
        max_val = np.max(mean_blurred)
        min_val = np.min(mean_blurred)
        mask = np.where(mean_blurred < min_val + (max_val - min_val) / 4)
        for img in images:
            blurred = gaussian_filter(img, sigma=7)
            img[mask] = blurred[mask]

    final_images, final_files = [], []

    if remove_low_contrast:
        for i, img in enumerate(images):
            Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
            # compute min and max of Y
            min_Y = np.min(Y)
            max_Y = np.max(Y)

            # compute contrast
            contrast = (max_Y - min_Y) / (int(max_Y) + int(min_Y))
            print(contrast, files[i])
            if contrast > CONTRAST_THRESHOLD:
                final_images.append(img)
                final_files.append(files[i])
    else:
        final_images = images
        final_files = files

    return final_images, final_files


def normalize_images_batched(inpath, outpath, bsize):
    path = Path(outpath)
    if path.exists():
        delete = input('Folder already exists. Delete? [y/n]:')
        if bool(delete):
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)

    loader = ImgLoader(inpath, batch_size=int(bsize))
    loop = tqdm(enumerate(loader), total=len(loader))

    for batch, (images, files) in loop:
        normalized_images, files = normalize_images(images, files)
        #save_images(normalized_images, out_path=outpath, file_names=files)
