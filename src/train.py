import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
import inquirer
# from datetime import datetime

from transforms import transform
from cityscapes_dataset import CustomCityscapesDataset
from model import CS_UNET
from utils import save_checkpoint, load_checkpoint


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
ROOT_DATA_DIR = '../data'
DATASET_NAME = 'Cityscapes'


def show_img_and_pred(img, target=None, prediction=None):
    fig = plt.figure()

    total_plots = sum(x is not None for x in [img, target, prediction])
    i = 1

    # show image
    ax = plt.subplot(1, total_plots, i)
    plt.title('Image')
    plt.imshow(img[0].permute(1, 2, 0))
    i += 1
    # show mask
    if target is not None:
        ax = plt.subplot(1, total_plots, i)
        plt.title('Target')
        plt.imshow(target[0])
        i += 1
    # show prediction
    if prediction is not None:
        ax = plt.subplot(1, total_plots, i)
        plt.title('Prediction')
        plt.imshow(prediction[0])
        i += 1
    plt.show()


def train_loop(loader, model, optimizer, loss_fn, writer=None, step=0):
    size = len(loader.dataset)
    for batch, (X, y) in enumerate(loader):
        X = X.float().to(DEVICE)
        # y is still long for some reason
        y = y.to(DEVICE)
        print(y.min(), y.max(), torch.unique(y))

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        jaccard = torchmetrics.JaccardIndex(len(loader.dataset.classes), ignore_index=255).to(DEVICE)
        jaccard_idx = jaccard(pred, y)

        if writer is not None:
            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            writer.add_scalar('Training Jaccard Index', jaccard_idx, global_step=step)

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        step += 1

    return step


def val_loop(loader, model, loss_fn):
    pass


def main():

    if DEVICE != 'cuda':
        questions = [inquirer.Confirm(name='proceed', message="Cuda Device not found. Proceed anyway?", default=False)]
        answers = inquirer.prompt(questions)
        if not answers['proceed']:
            exit()

    run_name = f"basic_no_aug_SGD" # _{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}
    run_dir = f"../runs/{DATASET_NAME}/{run_name}"
    run_file = f"{run_dir}/model.pth.tar"

    current_dataset = DATASET_NAME
    data_dir = f"{ROOT_DATA_DIR}/{current_dataset}"

    train_data = CustomCityscapesDataset(data_dir, transforms=transform)
    test_data = CustomCityscapesDataset(data_dir, mode='val', transforms=transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    print(len(train_data.classes))
    model = CS_UNET(in_ch=3, out_ch=len(train_data.classes)).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    step = 0
    epoch_global = 0
    if LOAD_MODEL:
        step, epoch_global = load_checkpoint(run_file, model, optimizer)

    print("\nBeginning Training\n")

    # logging
    writer = SummaryWriter(log_dir=run_dir)

    for epoch in range(NUM_EPOCHS):
        epoch_global += 1
        print(f"Epoch {epoch + 1}\n-------------------------------")
        step = train_loop(loader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn, writer=writer, step=step)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": step,
            "epochs": epoch_global
        }
        save_checkpoint(checkpoint, run_file)

    print("\nTraining Complete.")


if __name__ == '__main__':
    main()