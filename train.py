import random
from os import path

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

from src.masks import CowMaskGenerator
from src.transforms import transforms_train, transforms_train_mt, transforms_train_mt_basic, transforms_val, \
    transforms_generator
from src.datasets import CustomCityscapesDataset, VapourData, NO_LABEL
from src.model import CS_UNET, UnetResEncoder
from src.utils import save_checkpoint, load_checkpoint, IoU, alert_training_end, generate_pseudo_labels
from src.losses import cross_entropy_cons_loss
from src.lib.mean_teacher.data import TwoStreamBatchSampler
from src.lib.mean_teacher.losses import softmax_mse_loss
from src.lib.mean_teacher.ramps import sigmoid_rampup

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
BATCH_SIZE_UNLABELED = 12
NUM_EPOCHS = 600
NUM_WORKERS = 4
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MIN_DELTA = 1e-4
ES_PATIENCE = 30
LR_PATIENCE = 5
LRS_FACTOR = 0.1
LRS_ENABLED = True
PIN_MEMORY = True
CONTINUE = False
LOAD_PATH = None
CONSISTENCY = 1.0
CONSISTENCY_RAMPUP_LENGTH = 15
ROOT_DATA_DIR = './data'
DATASET_NAME = 'Cityscapes'
MT_ENABLED = True
EMA_DECAY = 0.996
MT_DELAY = 10
DROPOUT = None
DROPOUT_TEACHER = None
GENERATOR = None
CONS_LS_ON_LABELED_SAMPLES = True
DEV = True
USE_COWMASK = True
USE_ITERATIVE = True


def collate_split_batches(batch):
    """
    Collates lists of labeled and unlabeled samples into separate sub batches.
    Unlabeled samples are identified by their targets only containing NO_LABEL as values.

    :param batch: A list of (input, target) tuples
    :return: labeled inputs, labels, unlabeled inputs
    """
    input_labeled, input_unlabeled, labels = [], [], []
    for (input, target) in batch:
        if (target == NO_LABEL).all():
            input_unlabeled.append(input)
        else:
            input_labeled.append(input)
            labels.append(target)

    return torch.stack(input_labeled), torch.stack(labels), torch.stack(input_unlabeled) if input_unlabeled else None


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    :param worker_id: worker_id [0, ..., NUM_WORKER - 1]
    :return:
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seeders():
    """
    Sets all relevant seeds to preserver reproducibility between experiments.
    :return:
    """
    global GENERATOR
    random.seed(0)
    np.random.seed(0)
    GENERATOR = torch.Generator()
    GENERATOR.manual_seed(0)


def update_teacher_params(student_model: nn.Module, teacher_model: nn.Module, alpha, global_step):
    """
    Update the parameters of the teacher model when using the Mean Teacher semi-supervised learning approach. Code authored by CuriousAI.
    https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/main.py#L189

    :param student_model: the student model
    :param teacher_model: the teacher model, which is an ema of the student
    :param alpha: smoothing coefficient
    :param global_step: the overall steps the model has been trained for
    :return:
    """
    # ramp up alpha over the course of training
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # iterate through param dicts
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=(1 - alpha))
    # iterate through buffer dicts
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        # Doing it the same way as params produces error saying it can't be cast to float, even though the buffers previous type is float
        teacher_buffer.data = torch.mul(teacher_buffer.data, alpha).add_(student_buffer.data, alpha=(1 - alpha))


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


def apply_masks(masks, inputs, targets):
    mixed_inputs, mixed_labels = [], []
    for i in range(0, len(inputs), 2):
        mixed_inputs.append(
            inputs[i] * masks[int(i / 2)] + (1 - masks[int(i / 2)]) * inputs[i + 1])
        mixed_labels.append(targets[i] * torch.squeeze(masks[int(i / 2)], dim=1) + (
                1 - torch.squeeze(masks[int(i / 2)], dim=1)) * targets[i + 1])
    print(len(mixed_inputs), len(inputs))
    return torch.stack(mixed_inputs).to(DEVICE), torch.squeeze(torch.stack(mixed_labels)).to(DEVICE)


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, loss_fn, run_name, run_dir, scheduler=None,
                 teacher=None, consistency_fn=None, mask_loader=None, step=0, epoch_global=0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.run_name = run_name
        self.run_dir = run_dir
        self.run_file = f"{run_dir}/model.pth.tar"
        self.teacher = teacher
        self.consistency_fn = consistency_fn
        self.step = step
        self.epoch_global = epoch_global
        self.mask_loader = mask_loader
        self.best_loss = None
        self.best_iou = None
        self.patience_counter = 0

        # logging
        self.writer = SummaryWriter(log_dir=run_dir)

    def train(self):
        for epoch in range(NUM_EPOCHS):
            self.epoch_global += 1
            print(f"Epoch {epoch + 1} ({self.epoch_global})\n-------------------------------")
            self.train_epoch()
            save_checkpoint(self.model, teacher_model=self.teacher, optimizer=self.optimizer, scheduler=self.scheduler,
                            epoch_global=self.epoch_global, filename=self.run_file)

            if self.teacher is not None:
                losses, ious = val_fn(self.val_loader, self.teacher, self.loss_fn, self.epoch_global, self.writer,
                                      writer_suffix='Teacher')
                val_fn(self.val_loader, self.model, self.loss_fn, self.epoch_global, self.writer,
                       writer_suffix='Student')
            else:
                losses, ious = val_fn(self.val_loader, self.model, self.loss_fn, self.epoch_global, self.writer)

            val_loss = np.array(losses).sum() / len(losses)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            # early stopping
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_iou = np.array(ious).sum() / len(ious)
            elif self.best_loss - val_loss > MIN_DELTA:
                self.patience_counter = 0
                self.best_loss = val_loss
                self.best_iou = np.array(ious).sum() / len(ious)
                save_checkpoint(self.model, teacher_model=self.teacher, optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                epoch_global=self.epoch_global, filename=f"{self.run_dir}/model_best.pth.tar")
            else:
                self.patience_counter += 1
                print(
                    f"No validation loss improvement since {self.patience_counter} epochs.\nStopping after another {ES_PATIENCE - self.patience_counter} epochs without improvement.")

            if self.patience_counter >= ES_PATIENCE:
                print("Stopping early because of stagnant validation loss.")
                break

            print(f"-------------------------------\n")

        print("\nTraining Complete.")
        self.writer.add_hparams(
            {'lr': LEARNING_RATE, 'bsize': BATCH_SIZE, "lrs_factor": LRS_FACTOR, "lr_patience": LR_PATIENCE},
            {'hparams/loss': self.best_loss, 'hparams/iou': self.best_iou}, run_name='.')

        alert_training_end(self.run_name, self.epoch_global, stopped_early=(self.patience_counter >= ES_PATIENCE),
                           final_metrics={'best_loss': self.best_loss, 'best_iou': self.best_iou})

    def train_epoch(self):
        losses = []
        class_losses = []
        consistency_losses = []
        ious = []
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        skip_teacher = self.epoch_global <= MT_DELAY

        for batch, (input_labeled, target, input_unlabeled) in loop:

            # input is still long for some reason
            input_labeled = input_labeled.float().to(DEVICE)
            if input_unlabeled is not None and not skip_teacher:
                input_unlabeled = input_unlabeled.float().to(DEVICE)
            target = target.to(DEVICE)

            consistency_loss = 0

            # Supervised Learning
            pred_stu_labeled = self.model(input_labeled)
            class_loss = self.loss_fn(pred_stu_labeled, target)
            loss = class_loss

            # Unsupervised learning
            if MT_ENABLED and self.teacher is not None and not skip_teacher:
                # Calc teacher predictions
                pred_tch_unlabeled = self.teacher(input_unlabeled)
                pred_tch_labeled = self.teacher(input_labeled) if CONS_LS_ON_LABELED_SAMPLES else None
                print(input_unlabeled.size())
                if self.mask_loader is not None:
                    masks = next(self.mask_loader).to(DEVICE)
                    input_unlabeled, pred_tch_unlabeled = apply_masks(masks, input_unlabeled, pred_tch_unlabeled)
                # Unlabeled student predictions
                pred_stu_unlabeled = self.model(input_unlabeled)
                # Calc consistency loss
                consistency_loss_unlabeled = self.consistency_fn(pred_stu_unlabeled, pred_tch_unlabeled)
                consistency_loss_labeled = self.consistency_fn(pred_stu_labeled,
                                                          pred_tch_labeled) if CONS_LS_ON_LABELED_SAMPLES else 0

                # calculate losses depending on labeled or unlabeled samples
                consistency_weight = get_current_consistency_weight(self.epoch_global - MT_DELAY)
                consistency_loss = consistency_loss_labeled + consistency_loss_unlabeled
                loss = consistency_weight * consistency_loss + class_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.teacher is not None:
                update_teacher_params(self.model, self.teacher, EMA_DECAY, self.step)

            jaccard_idx, scores = IoU(pred=torch.argmax(nn.functional.softmax(pred_stu_labeled, 1), 1),
                                      ground_truth=target,
                                      n_classes=len(self.train_loader.dataset.classes))

            losses.append(float(loss.item()))
            class_losses.append(float(class_loss.item()))
            if self.teacher is not None and not skip_teacher:
                consistency_losses.append(float(consistency_loss.item()))
            ious.append(jaccard_idx)

            loop.set_postfix(loss=loss, jcc_idx=jaccard_idx)
            self.step += 1

        if self.writer is not None:
            self.writer.add_scalar('Training/Loss', np.array(losses).sum() / len(losses), global_step=self.epoch_global)
            self.writer.add_scalar('Training/Jaccard Index', np.array(ious).sum() / len(ious),
                                   global_step=self.epoch_global)
            if self.teacher is not None and not skip_teacher:
                self.writer.add_scalar('Training/Consistency Loss', np.array(consistency_losses).sum() / len(ious),
                                       global_step=self.epoch_global)
                self.writer.add_scalar('Training/Consistency Weight', consistency_weight, global_step=self.epoch_global)
            self.writer.add_scalar('Training/Class Loss', np.array(class_losses).sum() / len(ious),
                                   global_step=self.epoch_global)
            self.writer.add_scalar('Training/Learning Rate', self.optimizer.param_groups[0]['lr'],
                                   global_step=self.epoch_global)


def val_fn(loader, model, loss_fn, epoch=0, writer=None, writer_suffix=''):
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
            losses.append(loss.item())
            ious.append(jaccard_idx)

    if writer is not None:
        writer.add_scalar(f'Validation/Loss{" " + writer_suffix if writer_suffix != "" else ""}',
                          np.array(losses).sum() / len(losses), global_step=epoch)
        writer.add_scalar(f'Validation/Jaccard Index{" " + writer_suffix if writer_suffix != "" else ""}',
                          np.array(ious).sum() / len(ious), global_step=epoch)

    model.train()

    return losses, ious


def get_cs_loaders(data_dir, lbl_range, unlbl_range):
    train_data = CustomCityscapesDataset(data_dir, transforms=transforms_train, low_res=True, use_labeled=lbl_range,
                                         use_unlabeled=unlbl_range)
    val_data = CustomCityscapesDataset(data_dir, mode='val', transforms=transforms_val, low_res=True)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY,
                                  worker_init_fn=seed_worker if DEV else None, generator=GENERATOR, collate_fn=collate_split_batches)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY,
                                worker_init_fn=seed_worker if DEV else None, generator=GENERATOR)

    return train_dataloader, val_dataloader


def get_cs_loaders_mt(data_dir, lbl_range, unlbl_range):
    train_data = CustomCityscapesDataset(data_dir, transforms=transforms_train_mt_basic, low_res=True,
                                         use_labeled=lbl_range,
                                         use_unlabeled=unlbl_range)
    val_data = CustomCityscapesDataset(data_dir, mode='val', transforms=transforms_val, low_res=True)

    sampler = TwoStreamBatchSampler(train_data.unlabeled_idxs, train_data.labeled_idxs, batch_size=BATCH_SIZE,
                                    secondary_batch_size=BATCH_SIZE - BATCH_SIZE_UNLABELED)
    train_dataloader = DataLoader(train_data, pin_memory=PIN_MEMORY, batch_sampler=sampler,
                                  worker_init_fn=seed_worker if DEV else None, generator=GENERATOR,
                                  collate_fn=collate_split_batches)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY,
                                worker_init_fn=seed_worker if DEV else None, generator=GENERATOR)

    return train_dataloader, val_dataloader


def get_vap_loaders(data_dir, lbl_range, unlbl_range):
    train_data = VapourData(data_dir, transforms=transforms_train, use_labeled=lbl_range, use_unlabeled=unlbl_range)
    val_data = VapourData(data_dir, mode='val', transforms=transforms_val)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY,
                                  num_workers=NUM_WORKERS, worker_init_fn=seed_worker if DEV else None,
                                  generator=GENERATOR)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY,
                                worker_init_fn=seed_worker if DEV else None, generator=GENERATOR)

    return train_dataloader, val_dataloader


def get_vap_loaders_mt(data_dir, lbl_range, unlbl_range):
    train_data = VapourData(data_dir, transforms=transforms_train, use_labeled=lbl_range, use_unlabeled=unlbl_range)
    val_data = VapourData(data_dir, mode='val', transforms=transforms_val)

    train_dataloader = DataLoader(train_data, shuffle=True, pin_memory=PIN_MEMORY,
                                  batch_sampler=TwoStreamBatchSampler(train_data.unlabeled_idxs,
                                                                      train_data.labeled_idxs, batch_size=BATCH_SIZE,
                                                                      secondary_batch_size=BATCH_SIZE - BATCH_SIZE_UNLABELED),
                                  worker_init_fn=seed_worker if DEV else None, generator=GENERATOR,
                                  collate_fn=collate_split_batches)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY,
                                worker_init_fn=seed_worker if DEV else None, generator=GENERATOR)

    return train_dataloader, val_dataloader


def string_to_slice(string):
    parts = string.split('-')
    parts = [int(p) if len(p) > 0 else None for p in parts]
    return slice(parts[0], parts[1])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return CONSISTENCY * sigmoid_rampup(epoch, CONSISTENCY_RAMPUP_LENGTH)


def main():
    if DEV:
        set_seeders()

    parser = argparse.ArgumentParser(description="Start Model Training.")

    parser.add_argument("-rn", "--runname", help="Use a specific name for the run")
    parser.add_argument("-enc", "--encoder", help="Name of the timm model to use as the encoder")
    parser.add_argument("-lr", help="Set the initial learning rate")
    parser.add_argument("-bs", help="Set the batch size")
    parser.add_argument("-lrsp", help="Set the Patience for the learning rate scheduler")
    parser.add_argument("-lrsf", help="Set the Factor used to reduce the learning rate")
    parser.add_argument("-mf", help="Load from non default file")
    parser.add_argument("-lblr", "--labeledrange", help="Use this range of the dataset as labeled samples.",
                        type=string_to_slice)
    parser.add_argument("-ulblr", "--unlabeledrange", help="Use this range of the dataset as unlabeled samples.",
                        type=string_to_slice)
    parser.add_argument("--lrs", help="Set if learning rate scheduler should be used",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--mt", help="Set if mean teacher should be used",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--dropout", help="Set model dropout rate")
    parser.add_argument("--dropout_tch", help="Set teacher dropout rate when using MT")
    parser.add_argument("--mtdelay",
                        help="Set a number of epochs to train only on labeled data before mean teacher sets in")
    parser.add_argument("--iter", help="Use iterative semi supervised learning approach.",
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    global LEARNING_RATE
    global BATCH_SIZE
    global LR_PATIENCE
    global LRS_FACTOR
    global LRS_ENABLED
    global LOAD_PATH
    global DROPOUT
    global DROPOUT_TEACHER
    global MT_ENABLED
    global MT_DELAY
    global USE_ITERATIVE
    label_rng = None
    unlabel_rng = None

    if args.lr is not None:
        LEARNING_RATE = float(args.lr)
    if args.bs is not None:
        BATCH_SIZE = int(args.bs)
    if args.lrsp is not None:
        LR_PATIENCE = int(args.lrsp)
    if args.lrsf is not None:
        LRS_FACTOR = float(args.lrsf)
    if args.mf is not None:
        LOAD_PATH = args.mf
    if args.labeledrange is not None:
        label_rng = args.labeledrange
    if args.unlabeledrange is not None:
        unlabel_rng = args.unlabeledrange
    if args.lrs is not None:
        LRS_ENABLED = args.lrs
    if args.mt is not None:
        MT_ENABLED = args.mt
    if args.dropout is not None:
        DROPOUT = float(args.dropout)
    if args.dropout_tch is not None:
        DROPOUT_TEACHER = float(args.dropout_tch)
    if args.mtdelay is not None:
        MT_DELAY = int(args.mtdelay)
    if args.iter is not None:
        USE_ITERATIVE = args.iter

    if DEVICE != 'cuda':
        questions = [inquirer.Confirm(name='proceed', message="Cuda Device not found. Proceed anyway?", default=False)]
        answers = inquirer.prompt(questions)
        if not answers['proceed']:
            exit()

    run_name = (
        f"{args.runname or 'test'}_lrsp_{LR_PATIENCE}_lrsf_{LRS_FACTOR}_bs_{BATCH_SIZE}_lr_{LEARNING_RATE}_p_{ES_PATIENCE}_"
        + (
            f"cons_{CONSISTENCY}_cramp_{CONSISTENCY_RAMPUP_LENGTH}_mtd_{MT_DELAY}_emad_{EMA_DECAY}_" if MT_ENABLED else '')
        + f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
        if not CONTINUE else
        f"{args.runname or 'test'}")
    run_dir = f"./runs/{DATASET_NAME}/{run_name}"
    run_file = f"{run_dir}/model.pth.tar"

    current_dataset = DATASET_NAME
    data_dir = f"{ROOT_DATA_DIR}/{current_dataset}"

    # train_loader, val_loader = get_vap_loaders(data_dir, nr_to_use)
    train_loader, val_loader = (get_cs_loaders_mt(data_dir,
                                                  lbl_range=label_rng if label_rng is not None else slice(None, None),
                                                  unlbl_range=unlabel_rng if unlabel_rng is not None else slice(0, 0))
                                if MT_ENABLED else
                                get_cs_loaders(data_dir=data_dir,
                                               lbl_range=label_rng if label_rng is not None else slice(None, None),
                                               unlbl_range=slice(0, 0)))

    out_ch = len(train_loader.dataset.classes)

    print(out_ch)

    model = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d', dropout_p=DROPOUT).to(
        DEVICE)
    teacher = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d',
                             dropout_p=DROPOUT_TEACHER).to(DEVICE) if MT_ENABLED or USE_ITERATIVE else None

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    consistency_loss_fn = cross_entropy_cons_loss
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=LR_PATIENCE, threshold=MIN_DELTA,
                                                     threshold_mode='abs', verbose=True, factor=LRS_FACTOR,
                                                     cooldown=(ES_PATIENCE - LR_PATIENCE)) if LRS_ENABLED else None

    if USE_COWMASK:
        cow_mask_dataset = CowMaskGenerator(crop_size=(IMAGE_HEIGHT, IMAGE_WIDTH), method="mix")
        cow_mask_loader = DataLoader(dataset=cow_mask_dataset,
                                     batch_size=int(BATCH_SIZE_UNLABELED / 2),
                                     num_workers=NUM_WORKERS,
                                     worker_init_fn=seed_worker)
        cow_mask_iter = iter(cow_mask_loader)
    else:
        cow_mask_iter = None

    step = 0
    epoch_global = 0
    if CONTINUE:
        step, epoch_global = load_checkpoint(run_file, model, teacher_model=teacher, optimizer=optimizer,
                                             scheduler=scheduler)
    elif LOAD_PATH is not None:
        load_checkpoint(LOAD_PATH, model, except_layers=['final.weight', 'final.bias'], strict=False)

    if not USE_ITERATIVE:
        trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, run_name, run_dir, scheduler=scheduler, teacher=teacher, consistency_fn=consistency_loss_fn, mask_loader=cow_mask_iter, step=step, epoch_global=epoch_global)
        print("\nBeginning Training\n")
        trainer.train()
    else:
        # Supervised learning cycle
        print("\nStarting supervised training step.\n")
        trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, f"{run_name}_supervised_1", scheduler=scheduler, teacher=teacher, run_dir=path.join(run_dir, "supervised1"), step=step, epoch_global=epoch_global)
        trainer.train()

        # Generate pseudo labels
        generator_set = CustomCityscapesDataset(root_dir=data_dir, transforms=transforms_generator, use_labeled=slice(0, 0), use_unlabeled=unlabel_rng)
        generator_loaded = DataLoader(generator_set, batch_size=1, shuffle=False)
        output_path = path.join(data_dir, f"pseudo_labels_1_{run_name}")
        print("\nGenerate pseudo labels.\n")
        generate_pseudo_labels(model=teacher, loader=generator_loaded, output_dir=output_path, device=DEVICE)

        # Train with pseudo labels
        model = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d', dropout_p=DROPOUT).to(
            DEVICE)
        teacher = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d',
                                 dropout_p=DROPOUT_TEACHER).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=LR_PATIENCE, threshold=MIN_DELTA,
                                                         threshold_mode='abs', verbose=True, factor=LRS_FACTOR,
                                                         cooldown=(ES_PATIENCE - LR_PATIENCE)) if LRS_ENABLED else None
        train_set = CustomCityscapesDataset(root_dir=data_dir, transforms=transforms_train, use_pseudo_labels=unlabel_rng, pseudo_label_dir=output_path)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker, generator=GENERATOR, shuffle=True)
        trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, f"{run_name}_pseudo_1", scheduler=scheduler, teacher=teacher, run_dir=path.join(run_dir, "pseudo1"))
        print("\nTrain on pseudo labels.\n")
        trainer.train()

        # Fine Tune with labeled data
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=LR_PATIENCE, threshold=MIN_DELTA,
                                                         threshold_mode='abs', verbose=True, factor=LRS_FACTOR,
                                                         cooldown=(ES_PATIENCE - LR_PATIENCE)) if LRS_ENABLED else None
        train_set = CustomCityscapesDataset(root_dir=data_dir, transforms=transforms_train, use_labeled=label_rng)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker, generator=GENERATOR, shuffle=True)
        trainer = Trainer(model, optimizer, train_loader, val_loader, loss_fn, f"{run_name}_tune_1", scheduler=scheduler, teacher=teacher, run_dir=path.join(run_dir, "tune1"))
        print("\nFine tune on labeled data.\n")
        trainer.train()


if __name__ == '__main__':
    main()
