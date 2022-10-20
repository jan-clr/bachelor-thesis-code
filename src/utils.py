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


load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def save_checkpoint(model, optimizer=None, scheduler=None, step=0, epoch_global=0, filename="my_checkpoint.pth.tar") -> None:# save model
    checkpoint = {
	    "state_dict": model.state_dict(),
        "steps": step,
        "epochs": epoch_global
                  }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file: str, model, optimizer=None, scheduler=None) -> (int, int):
    """
    Loads the models (and optimizers) parameters from a checkpoint file.

    :param scheduler: The scheduler whose parameters to load if not None
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
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

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


def main():
    alert_training_end('testrun', 10, stopped_early=False)


if __name__ == '__main__':
    main()