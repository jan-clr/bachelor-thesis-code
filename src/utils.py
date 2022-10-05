import torch
import cv2
import os
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


def save_checkpoint(state, filename="my_checkpoint.pth.tar") -> None:
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file: str, model, optimizer=None) -> (int, int):
    """
    Loads the models (and optimizers) parameters from a checkpoint file.

    :param checkpoint_file: The path of the checkpoint file
    :param model: The model whose parameters to load
    :param optimizer: The optimizer whose parameters to load if not None
    :return: (steps, epochs) : The total steps and epochs in the models training
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint['steps'], checkpoint["epochs"]


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


def IoU(pred: torch.Tensor, ground_truth: torch.Tensor, n_classes:int, ignore_ids: [int]=[]) -> (float, any) :
    """
    Compute the Jaccard Index for a prediciton given a ground truth.

    Based on https://stackoverflow.com/a/48383182/10540901

    :param pred: The predictions of the model
    :param ground_truth: The target to test against
    :param n_classes: number of consecutive class ids. Ids outside this range are ignored.
    :param ignore_ids: array of ids to ignore for evaluation.
    :return: The average IoU score, and the score for each evaluated class
    """

    if pred.shape != ground_truth.shape:
        raise ValueError(f"Prediction and ground truth must have same shape, but shape {pred.shape} and {ground_truth.shape} were given.")

    ious = []
    pred = pred.view(-1)
    target = ground_truth.view(-1)

    # Ignore IoU for values in ignore index
    for cls in [x for x in range(0, n_classes) if x not in ignore_ids]:
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    include_in_eval = np.array([score for score in ious if not np.isnan(score)])

    SMOOTH = 1e-6
    avg = (include_in_eval.sum() + SMOOTH) / (len(include_in_eval) + SMOOTH)

    return avg, np.array(ious)


def resize_images(from_path, to_path, size, anti_aliasing=True):
    # recreate dir structure
    Path(to_path).mkdir(parents=True, exist_ok=True)

    src_prefix = len(from_path) + len(os.path.sep)

    for root, dirs, files in os.walk(from_path):
        for dirname in dirs:
            dirpath = os.path.join(to_path, root[src_prefix:], dirname)
            Path(dirpath).mkdir(exist_ok=True)

    images = glob(f"{from_path}/**/*.png", recursive=True)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    i = 0
    for img_file in images:
        img = Image.open(img_file)
        resized = TF.resize(img, size,
                            interpolation=InterpolationMode.BILINEAR if anti_aliasing else InterpolationMode.NEAREST)
        resized.save(os.path.join(to_path, img_file[src_prefix:]))

        i += 1
        print(f"\rResized {i}/{len(images)}", end='')