import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from tqdm import tqdm
import inquirer
import argparse
from datetime import datetime
import gc

from transforms import transforms_train, transforms_val
from datasets import CustomCityscapesDataset, VapourData
from model import CS_UNET, UnetResEncoder
from utils import save_checkpoint, load_checkpoint, IoU, alert_training_end

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 400
NUM_WORKERS = 1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MIN_DELTA = 1e-4
ES_PATIENCE = 40
LR_PATIENCE = 5
LRS_FACTOR = 0.1
PIN_MEMORY = True
CONTINUE = False
LOAD_PATH = None
ROOT_DATA_DIR = '../data'
DATASET_NAME = 'Cityscapes'


def show_img_and_pred(img, target=None, prediction=None):
    fig = plt.figure()

    total_plots = sum(x is not None for x in [img, target, prediction])
    i = 1

    # show
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
    losses = []
    ious = []
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch, (X, y) in loop:
        X = X.float().to(DEVICE)
        # y is still long for some reason
        y = y.to(DEVICE)
        # print(y.min(), y.max(), torch.unique(y))

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # jaccard = torchmetrics.JaccardIndex(len(loader.dataset.classes), ignore_index=255).to(DEVICE)
        # jaccard_idx = jaccard(pred, y)

        jaccard_idx, scores = IoU(pred=torch.argmax(nn.functional.softmax(pred, 1), 1), ground_truth=y,
                                  n_classes=len(loader.dataset.classes))

        del X, y

        losses.append(float(loss.item()))
        ious.append(jaccard_idx)

        # if batch % 20 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        loop.set_postfix(loss=loss, jcc_idx=jaccard_idx)

    if writer is not None:
        writer.add_scalar('Training/Loss', np.array(losses).sum() / len(losses), global_step=step)
        writer.add_scalar('Training/Jaccard Index', np.array(ious).sum() / len(ious), global_step=step)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step=step)

    return losses, ious


def val_fn(loader, model, loss_fn, step=0, writer=None):
    model.eval()
    losses = []
    ious = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(loader):
            X = X.float().to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)
            loss = loss_fn(pred, y)
            jaccard_idx, scores = IoU(pred=torch.argmax(nn.functional.softmax(pred.float(), 1), 1), ground_truth=y,
                                      n_classes=len(loader.dataset.classes))

            del X, y
            losses.append(loss.item())
            ious.append(jaccard_idx)

    if writer is not None:
        writer.add_scalar('Validation/Loss', np.array(losses).sum() / len(losses), global_step=step)
        writer.add_scalar('Validation/Jaccard Index', np.array(ious).sum() / len(ious), global_step=step)

    model.train()

    return losses, ious


def get_cs_loaders(data_dir):
    train_data = CustomCityscapesDataset(data_dir, transforms=transforms_train, low_res=False)
    val_data = CustomCityscapesDataset(data_dir, mode='val', transforms=transforms_val, low_res=False)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    return train_dataloader, val_dataloader


def get_vap_loaders(data_dir, nr_to_use):
    train_data = VapourData(data_dir, transforms=transforms_train, nr_to_use=nr_to_use)
    val_data = VapourData(data_dir, mode='val', transforms=transforms_val)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Start Model Training.")

    parser.add_argument("-rn", "--runname", help="Use a specific name for the run")
    parser.add_argument("-enc", "--encoder", help="Name of the timm model to use as the encoder")
    parser.add_argument("-lr", help="Set the initial learning rate")
    parser.add_argument("-bs", help="Set the batch size")
    parser.add_argument("-lrsp", help="Set the Patience for the learning rate scheduler")
    parser.add_argument("-lrsf", help="Set the Factor used to reduce the learning rate")
    parser.add_argument("-mf", help="Load from non default file")
    parser.add_argument("-red", help="Use a reduced number of samples from the dataset if possible")

    args = parser.parse_args()

    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    lr_patience = LR_PATIENCE
    lrs_factor = LRS_FACTOR
    model_to_load = LOAD_PATH
    nr_to_use = None

    if args.lr is not None:
        learning_rate = float(args.lr)
    if args.bs is not None:
        batch_size = int(args.bs)
    if args.lrsp is not None:
        lr_patience = int(args.lrsp)
    if args.lrsf is not None:
        lrs_factor = float(args.lrsf)
    if args.mf is not None:
        model_to_load = args.mf
    if args.red is not None:
        nr_to_use = int(args.red)

    if DEVICE != 'cuda':
        questions = [inquirer.Confirm(name='proceed', message="Cuda Device not found. Proceed anyway?", default=False)]
        answers = inquirer.prompt(questions)
        if not answers['proceed']:
            exit()

    run_name = f"{args.runname or 'test'}_lrsp_{lr_patience}_lrsf_{lrs_factor}_bs_{batch_size}_lr_{learning_rate}_p_{ES_PATIENCE}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')} "
    run_dir = f"../runs/{DATASET_NAME}/{run_name}"
    run_file = f"{run_dir}/model.pth.tar"

    current_dataset = DATASET_NAME
    data_dir = f"{ROOT_DATA_DIR}/{current_dataset}"

    # train_loader, val_loader = get_vap_loaders(data_dir, nr_to_use)
    train_loader, val_loader = get_cs_loaders(data_dir)

    out_ch = len(train_loader.dataset.classes)

    print(out_ch)

    model = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d').to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=lr_patience, threshold=MIN_DELTA, threshold_mode='abs', verbose=True, factor=lrs_factor, cooldown=(ES_PATIENCE - lr_patience))

    step = 0
    epoch_global = 0
    if CONTINUE:
        step, epoch_global = load_checkpoint(run_file, model, optimizer, scheduler)
    elif model_to_load is not None:
        load_checkpoint(model_to_load, model, except_layers=['final.weight', 'final.bias'], strict=False)
    print("\nBeginning Training\n")
    # logging
    writer = SummaryWriter(log_dir=run_dir)
    images = next(iter(train_loader))[0].to(DEVICE)
    writer.add_graph(model, images)
    best_loss = None
    best_iou = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_global += 1
        print(f"Epoch {epoch + 1} ({epoch_global})\n-------------------------------")
        train_loop(loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, writer=writer,
                   step=epoch_global)

        save_checkpoint(model, optimizer=optimizer, scheduler=scheduler, epoch_global=epoch_global, filename=run_file)
        # Try flushing GPU cache before validation for large datasets
        torch.cuda.empty_cache()
        gc.collect()

        # if epoch_global % 10 == 0:
        #     save_checkpoint(model, optimizer=optimizer, scheduler=scheduler, epoch_global=epoch_global, filename=f"{run_dir}/model_{epoch_global}.pth.tar")

        losses, ious = val_fn(val_loader, model, loss_fn, epoch_global, writer)
        val_loss = np.array(losses).sum() / len(losses)
        scheduler.step(val_loss)
        # early stopping
        if best_loss is None:
            best_loss = val_loss
            best_iou = np.array(ious).sum() / len(ious)
        elif best_loss - val_loss > MIN_DELTA:
            patience_counter = 0
            best_loss = val_loss
            best_iou = np.array(ious).sum() / len(ious)
            save_checkpoint(model, optimizer=optimizer, scheduler=scheduler, epoch_global=epoch_global, filename=f"{run_dir}/model_best.pth.tar")
        else:
            patience_counter += 1
            print(
                f"No validation loss improvement since {patience_counter} epochs.\nStopping after another {ES_PATIENCE - patience_counter} epochs without improvement.")

        if patience_counter >= ES_PATIENCE:
            print("Stopping early because of stagnant validation loss.")
            break

        print(f"-------------------------------\n")

    print("\nTraining Complete.")
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, "lrs_factor": lrs_factor, "lr_patience": lr_patience}, {'hparams/loss': best_loss, 'hparams/iou': best_iou}, run_name='.')

    alert_training_end(run_name, epoch_global, stopped_early=(patience_counter >= ES_PATIENCE), final_metrics={'best_loss' : best_loss, 'best_iou': best_iou})


if __name__ == '__main__':
    main()
