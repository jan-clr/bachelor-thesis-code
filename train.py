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

from src.transforms import transforms_train, transforms_train_mt, transforms_val
from src.datasets import CustomCityscapesDataset, VapourData
from src.model import CS_UNET, UnetResEncoder
from src.utils import save_checkpoint, load_checkpoint, IoU, alert_training_end
from src.losses import cross_entropy_cons_loss
from src.lib.mean_teacher.data import TwoStreamBatchSampler
from src.lib.mean_teacher.losses import softmax_mse_loss
from src.lib.mean_teacher.ramps import sigmoid_rampup

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 400
NUM_WORKERS = 1
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MIN_DELTA = 1e-4
ES_PATIENCE = 80
LR_PATIENCE = 5
LRS_FACTOR = 0.1
LRS_ENABLED = True
PIN_MEMORY = True
CONTINUE = False
LOAD_PATH = None
CONSISTENCY = 0.5
CONSISTENCY_RAMPUP_LENGTH = 100
ROOT_DATA_DIR = './data'
DATASET_NAME = 'Cityscapes'
MT_ENABLED = True
EMA_DECAY = 0.998
MT_DELAY = 10
DROPOUT = None
DROPOUT_TEACHER = None


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


def train_loop(loader, model, optimizer, loss_fn, writer=None, step=0, epoch=0):
	size = len(loader.dataset)
	losses = []
	ious = []
	loop = tqdm(enumerate(loader), total=len(loader), leave=False)
	steps = step

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

		losses.append(float(loss.item()))
		ious.append(jaccard_idx)

		# if batch % 20 == 0:
		#     loss, current = loss.item(), batch * len(X)
		#     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

		loop.set_postfix(loss=loss, jcc_idx=jaccard_idx)

	if writer is not None:
		writer.add_scalar('Training/Loss', np.array(losses).sum() / len(losses), global_step=epoch)
		writer.add_scalar('Training/Jaccard Index', np.array(ious).sum() / len(ious), global_step=epoch)
		writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], global_step=epoch)

	return losses, ious, steps


def train_loop_mt(loader, student_model, teacher_model, optimizer, loss_fn, consistency_fn, writer=None, step=0,
                  epoch=0):
	size = len(loader.dataset)
	losses = []
	class_losses = []
	consistency_losses = []
	ious = []
	loop = tqdm(enumerate(loader), total=len(loader), leave=False)
	steps = step
	skip_teacher = epoch <= MT_DELAY

	for batch, ((input_stu, input_tch), target) in loop:

		# input is still long for some reason
		input_stu = input_stu.float().to(DEVICE)
		if not skip_teacher:
			input_tch = input_tch.float().to(DEVICE)
		target = target.to(DEVICE)
		# print(y.min(), y.max(), torch.unique(y))

		# Compute predictions
		pred_stu = student_model(input_stu)
		pred_tch = teacher_model(input_tch) if not skip_teacher else None

		# calculate losses depending on labeled or unlabeled samples
		consistency_loss = consistency_fn(pred_stu, pred_tch) if not skip_teacher else 0
		consistency_weight = get_current_consistency_weight(epoch - MT_DELAY)
		class_loss = loss_fn(pred_stu, target)

		loss = consistency_weight * consistency_loss + class_loss
		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		update_teacher_params(student_model, teacher_model, EMA_DECAY, step)

		jaccard_idx, scores = IoU(pred=torch.argmax(nn.functional.softmax(pred_stu, 1), 1), ground_truth=target,
		                          n_classes=len(loader.dataset.classes))

		losses.append(float(loss.item()))
		class_losses.append(float(class_loss.item()))
		if not skip_teacher:
			consistency_losses.append(float(consistency_loss.item()))
		ious.append(jaccard_idx)

		loop.set_postfix(loss=loss, jcc_idx=jaccard_idx)
		step += 1

	if writer is not None:
		writer.add_scalar('Training/Loss', np.array(losses).sum() / len(losses), global_step=epoch)
		writer.add_scalar('Training/Jaccard Index', np.array(ious).sum() / len(ious), global_step=epoch)
		if not skip_teacher:
			writer.add_scalar('Training/Consistency Loss', np.array(consistency_losses).sum() / len(ious),
			                  global_step=epoch)
		writer.add_scalar('Training/Class Loss', np.array(class_losses).sum() / len(ious), global_step=epoch)
		writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], global_step=epoch)
		writer.add_scalar('Training/Consistency Weight', consistency_weight, global_step=epoch)

	return losses, ious, steps


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

	print(len(train_data))

	train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
	val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

	return train_dataloader, val_dataloader


def get_cs_loaders_mt(data_dir, lbl_range, unlbl_range):
	train_data = CustomCityscapesDataset(data_dir, transforms=transforms_train_mt, low_res=True, use_labeled=lbl_range,
	                                     use_unlabeled=unlbl_range)
	val_data = CustomCityscapesDataset(data_dir, mode='val', transforms=transforms_val, low_res=True)

	sampler = TwoStreamBatchSampler(train_data.labeled_idxs, train_data.unlabeled_idxs, batch_size=BATCH_SIZE,
	                                secondary_batch_size=3 * BATCH_SIZE // 4)
	train_dataloader = DataLoader(train_data, pin_memory=PIN_MEMORY, batch_sampler=sampler)
	val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

	return train_dataloader, val_dataloader


def get_vap_loaders(data_dir, lbl_range, unlbl_range):
	train_data = VapourData(data_dir, transforms=transforms_train, use_labeled=lbl_range, use_unlabeled=unlbl_range)
	val_data = VapourData(data_dir, mode='val', transforms=transforms_val)

	train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
	val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

	return train_dataloader, val_dataloader


def get_vap_loaders_mt(data_dir, lbl_range, unlbl_range):
	train_data = VapourData(data_dir, transforms=transforms_train, use_labeled=lbl_range, use_unlabeled=unlbl_range)
	val_data = VapourData(data_dir, mode='val', transforms=transforms_val)

	train_dataloader = DataLoader(train_data, shuffle=True, pin_memory=PIN_MEMORY,
	                              batch_sampler=TwoStreamBatchSampler(train_data.labeled_idxs,
	                                                                  train_data.unlabeled_idxs, batch_size=BATCH_SIZE,
	                                                                  secondary_batch_size=BATCH_SIZE // 4))
	val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

	return train_dataloader, val_dataloader


def string_to_slice(string):
	parts = string.split('-')
	parts = [int(p) if len(p) > 0 else None for p in parts]
	return slice(parts[0], parts[1])


def get_current_consistency_weight(epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	return CONSISTENCY * sigmoid_rampup(epoch, CONSISTENCY_RAMPUP_LENGTH)


def main():
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

	if DEVICE != 'cuda':
		questions = [inquirer.Confirm(name='proceed', message="Cuda Device not found. Proceed anyway?", default=False)]
		answers = inquirer.prompt(questions)
		if not answers['proceed']:
			exit()

	run_name = (
		f"{args.runname or 'test'}_lrsp_{LR_PATIENCE}_lrsf_{LRS_FACTOR}_bs_{BATCH_SIZE}_lr_{LEARNING_RATE}_p_{ES_PATIENCE}_"
		+ (f"cons_{CONSISTENCY}_cramp_{CONSISTENCY_RAMPUP_LENGTH}_mtd_{MT_DELAY}_emad_{EMA_DECAY}_" if MT_ENABLED else '')
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
	                                           unlbl_range=unlabel_rng if unlabel_rng is not None else slice(0, 0)))

	out_ch = len(train_loader.dataset.classes)

	print(out_ch)

	model = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d', dropout_p=DROPOUT).to(
		DEVICE)
	teacher = UnetResEncoder(in_ch=3, out_ch=out_ch, encoder_name=args.encoder or 'resnet34d',
	                         dropout_p=DROPOUT_TEACHER).to(DEVICE) if MT_ENABLED else None

	loss_fn = nn.CrossEntropyLoss(ignore_index=255)
	consistency_loss_fn = cross_entropy_cons_loss
	# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=LR_PATIENCE, threshold=MIN_DELTA,
	                                                 threshold_mode='abs', verbose=True, factor=LRS_FACTOR,
	                                                 cooldown=(ES_PATIENCE - LR_PATIENCE)) if LRS_ENABLED else None

	step = 0
	epoch_global = 0
	if CONTINUE:
		step, epoch_global = load_checkpoint(run_file, model, teacher_model=teacher, optimizer=optimizer,
		                                     scheduler=scheduler)
	elif LOAD_PATH is not None:
		load_checkpoint(LOAD_PATH, model, except_layers=['final.weight', 'final.bias'], strict=False)

	print("\nBeginning Training\n")
	# logging
	writer = SummaryWriter(log_dir=run_dir)
	# images = next(iter(train_loader))[0].to(DEVICE)
	# writer.add_graph(model, images)
	best_loss = None
	best_iou = None
	patience_counter = 0

	for epoch in range(NUM_EPOCHS):
		epoch_global += 1
		print(f"Epoch {epoch + 1} ({epoch_global})\n-------------------------------")
		_, _, step = (
			train_loop_mt(loader=train_loader, student_model=model, teacher_model=teacher, optimizer=optimizer,
			              loss_fn=loss_fn, consistency_fn=consistency_loss_fn, writer=writer,
			              step=step, epoch=epoch_global)
			if MT_ENABLED else
			train_loop(loader=train_loader,
			           model=model,
			           optimizer=optimizer,
			           loss_fn=loss_fn,
			           writer=writer,
			           step=step,
			           epoch=epoch_global))

		save_checkpoint(model, teacher_model=teacher, optimizer=optimizer, scheduler=scheduler,
		                epoch_global=epoch_global, filename=run_file)

		losses, ious = [], []
		if MT_ENABLED:
			losses, ious = val_fn(val_loader, teacher, loss_fn, epoch_global, writer, writer_suffix='Teacher')
			val_fn(val_loader, model, loss_fn, epoch_global, writer, writer_suffix='Student')
		else:
			losses, ious = val_fn(val_loader, model, loss_fn, epoch_global, writer)

		val_loss = np.array(losses).sum() / len(losses)
		if LRS_ENABLED:
			scheduler.step(val_loss)
		# early stopping
		if best_loss is None:
			best_loss = val_loss
			best_iou = np.array(ious).sum() / len(ious)
		elif best_loss - val_loss > MIN_DELTA:
			patience_counter = 0
			best_loss = val_loss
			best_iou = np.array(ious).sum() / len(ious)
			save_checkpoint(model, teacher_model=teacher, optimizer=optimizer, scheduler=scheduler,
			                epoch_global=epoch_global, filename=f"{run_dir}/model_best.pth.tar")
		else:
			patience_counter += 1
			print(
				f"No validation loss improvement since {patience_counter} epochs.\nStopping after another {ES_PATIENCE - patience_counter} epochs without improvement.")

		if patience_counter >= ES_PATIENCE:
			print("Stopping early because of stagnant validation loss.")
			break

		print(f"-------------------------------\n")

	print("\nTraining Complete.")
	writer.add_hparams({'lr': LEARNING_RATE, 'bsize': BATCH_SIZE, "lrs_factor": LRS_FACTOR, "lr_patience": LR_PATIENCE},
	                   {'hparams/loss': best_loss, 'hparams/iou': best_iou}, run_name='.')

	alert_training_end(run_name, epoch_global, stopped_early=(patience_counter >= ES_PATIENCE),
	                   final_metrics={'best_loss': best_loss, 'best_iou': best_iou})


if __name__ == '__main__':
	main()
