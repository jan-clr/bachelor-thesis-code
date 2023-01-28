import cv2
import torch
import os
import numpy as np
from pathlib import Path
from glob import glob
from PIL import Image, ImageFile
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from slack_sdk import WebhookClient
from dotenv import load_dotenv
import socket
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def show(imgs):
    """
    https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py
    :param imgs: 
    :return: 
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def save_checkpoint(model, teacher_model=None, optimizer=None, scheduler=None, step=0, epoch_global=0, filename="my_checkpoint.pth.tar") -> None:# save model
    checkpoint = {
	    "state_dict": model.state_dict(),
        "steps": step,
        "epochs": epoch_global
                  }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if teacher_model is not None:
        checkpoint["teacher_dict"] = teacher_model.state_dict()

    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file: str, model, teacher_model=None, optimizer=None, scheduler=None, strict=True, except_layers=[]) -> (int, int):
    """
    Loads the models (and optimizers) parameters from a checkpoint file.

    :param teacher_model: A teacher model to load when using Mean Teacher
    :param strict: Use strict loading
    :param scheduler: The scheduler whose parameters to load if not None
    :param checkpoint_file: The path of the checkpoint file
    :param model: The model whose parameters to load
    :param optimizer: The optimizer whose parameters to load if not None
    :return: (steps, epochs) : The total steps and epochs in the models training
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in except_layers}
    model.load_state_dict(pretrained_dict, strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if teacher_model is not None:
        teacher_model.load_state_dict(checkpoint["teacher_dict"])

    return checkpoint['steps'], checkpoint["epochs"]


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
            ious.append(float('nan'))  # If there is no ground truth and no prediction, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    include_in_eval = np.array([score for score in ious if not np.isnan(score)])

    SMOOTH = 1e-6
    avg = (include_in_eval.sum() + SMOOTH) / (len(include_in_eval) + SMOOTH)

    return avg, np.array(ious)


def perform_on_all_imgs(from_path, to_path, operations=None):
    """
    Performs a sequence of operations on all files found in a directory and saves the results to another directory.
    :param from_path: Source dir
    :param to_path: Target dir
    :param operations: List of operations. Applied in sequential order, so make sure output and input of consecutive ops match. Output must be torch tensor(-like)
    :return:
    """
    # recreate dir structure
    if operations is None:
        operations = []
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
        transformed = Image.open(img_file)
        for op in operations:
            transformed = op(transformed)

        transformed.save(os.path.join(to_path, img_file[src_prefix:]))

        i += 1
        print(f"\rTransformed {i}/{len(images)}", end='')


def resize_images(from_path, to_path, size, anti_aliasing=True):
    """
    Resizes all images in dir
    :param from_path:
    :param to_path:
    :param size:
    :param anti_aliasing:
    :return:
    """
    print("Resizing ----------------------")
    operations = [lambda img :TF.resize(img, size,
              interpolation=InterpolationMode.BILINEAR if anti_aliasing else InterpolationMode.NEAREST)]
    perform_on_all_imgs(from_path, to_path, operations)
    print('')


def split_images(from_path, to_path, file_ext='png', split_factor=2, filter_only_background=False):
    print("Splitting ---------------------")
    # recreate dir structure
    Path(to_path).mkdir(parents=True, exist_ok=True)

    src_prefix = len(from_path) + len(os.path.sep)

    for root, dirs, files in os.walk(from_path):
        for dirname in dirs:
            dirpath = os.path.join(to_path, root[src_prefix:], dirname)
            Path(dirpath).mkdir(exist_ok=True)

    images = glob(f"{from_path}/**/*.{file_ext}", recursive=True)

    i = 0
    for img_file in images:
        img = cv2.imread(img_file)
        splits = split_image(img, split_factor)
        for idx, crop in enumerate(splits):
            if filter_only_background and (crop == 0).all():
                continue
            subpath = Path(img_file[src_prefix:])
            cv2.imwrite(os.path.join(to_path, subpath.parent, f"{idx}_{subpath.name}"), crop)

        i += 1
        print(f"\rTransformed {i}/{len(images)}", end='')

    print('')


def split_image(image, split_factor):
    h, w, _ = image.shape if len(image.shape) == 3 else (image.shape[0], image.shape[1], 1)
    y_points = np.arange(0, h, h // split_factor)
    x_points = np.arange(0, w, w // split_factor)
    splits = []
    for i in range(split_factor):
        for j in range(split_factor):
            splits.append(image[y_points[i]: y_points[i] + h // split_factor, x_points[j]: x_points[j] + w // split_factor])

    return splits

def generate_pseudo_labels(model, loader, output_dir, device):
    path = Path(output_dir)
    path.mkdir(parents=True)
    model.eval()
    nr_samples = len(loader) * loader.batch_size
    digits = len(str(nr_samples))
    print(f"Generating pseudo labels for {nr_samples} images")
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (image, target) in loop:
        image = image.float().to(device)
        pred = model(image)
        pred = torch.squeeze(pred)
        label = torch.argmax(pred, dim=len(pred.size()) - 3) # size as 4 entries if images are batched
        if len(label.size()) == 3:
            # / 255.0 because torch io expects float tensors [0.0, 1.0]
            # replace with functionality that natively supports int
            for j in range(len(label)):
                save_image(label[i] / 255.0, os.path.join(output_dir, f"{i + j:0{digits}}.png"))
        else:
            save_image(label / 255.0, os.path.join(output_dir, f"{i:0{digits}}.png"))


def get_lowest_unused_int(l):
    """
    Returns the lowest non-negative integer that is not included in a list of integers
    :param l: A list of integers
    :return: the lowest uncontained integer >= 0
    """
    return min(set(range(max(l) + 2)) - set(l))


def send_slack_msg(content, text="Fallback Alert"):
    if SLACK_WEBHOOK_URL is None:
        print("No webhook url specified. Make sure .env is setup correctly.")
        return

    webhook = WebhookClient(SLACK_WEBHOOK_URL)
    response = webhook.send(
        text=text,
        blocks=content
    )


def alert_training_end(run_name, epoch=0, final_metrics=None, stopped_early=False):
    blocks = [{
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Training Complete"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"The training for the run *{run_name}* has been completed after *{epoch} epochs*, because {'improvement has stopped' if stopped_early else 'the maximum number of epochs has been reached'}."
            }
        },
        ]
    if final_metrics is not None:
        blocks.append({
            "type": "divider"
        })
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{final_metrics}"
            }
        })

    blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": f"Host: {socket.gethostname()}"
                }
            ]
        })
    send_slack_msg(blocks, "Training Complete")


class ImgLoader:
    def __init__(self, path, batch_size=1):
        if not os.path.isdir(path):
            print('Path is not a directory.')
        self.path = path
        self.image_files = sorted(os.listdir(path))
        self.batch_size = batch_size
        self.batches = int(np.ceil(len(self.image_files) / float(batch_size)))

    def __len__(self):
        return self.batches

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.batches - 1:
            batch_files = self.image_files[self.counter * self.batch_size: (self.counter + 1) * self.batch_size]
        elif self.counter == self.batches - 1:
            batch_files = self.image_files[self.counter * self.batch_size: None]
        else:
            raise StopIteration
        images, read_files = [], []
        for file in batch_files:
            cv_img = cv2.imread(os.path.join(self.path, file))
            if cv_img is not None:
                images.append(cv_img)
                read_files.append(file)
            else:
                print(f"{file} could not be read. Ignoring.")
        self.counter += 1
        return np.array(images), read_files


def save_images(images, out_path, file_names):
    assert len(images) == len(file_names)
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(out_path, file_names[i]), img)


def remove_where_mask_empty(impath, maskpath, empty_label=0):
    mask_files = glob(os.path.join(maskpath, '**/*labelIds*'), recursive=True)
    img_files = glob(os.path.join(impath, '**/*.*'), recursive=True)
    for mf in mask_files:
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if (mask == empty_label).all():
            mask_path = Path(mf)
            match_pattern = mask_path.name[:-len('gtFine_labelIds.png')] + 'leftImg8bit'
            matching_imgs = [fn for fn in img_files if match_pattern in fn]
            for img in matching_imgs:
                if os.path.isfile(img):
                    os.remove(img)
            if os.path.isfile(mf):
                os.remove(mf)


def create_dir(pathname):
    path = Path(pathname)
    if path.exists():
        delete = input('Folder already exists. Delete? [y/n]:')
        if delete == 'y':
            shutil.rmtree(path)
        else:
            exit()
    path.mkdir(parents=True, exist_ok=False)
