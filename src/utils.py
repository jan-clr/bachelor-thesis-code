import torch
import cv2
import os


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def normalize_data(path):
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
