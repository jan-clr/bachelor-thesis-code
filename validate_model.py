import torch
import argparse

from torch.utils.data import DataLoader

from src.datasets import VapourData
from src.model import UnetResEncoder
from src.transforms import transforms_val
from src.utils import load_checkpoint, val_fn
import csv
import os
import pathlib

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser("Validate a model for reporting.")

    parser.add_argument('--checkpoint', help='The path to a checkpoint file to load.', required=True)
    parser.add_argument('--modelname', help='The name for the model validation should be saved under.')
    parser.add_argument('--data', help='The path to a the validation dataset.', required=True)
    parser.add_argument('--outpath', help='The path to save the model to.', default='.')
    parser.add_argument('--split', help='The path to save the model to.', default=None, type=int)
    parser.add_argument('--bs', help='Batch size to use.', default=1, type=int)
    parser.add_argument('--oidx', help='Set the out indices for the feature extractor.', nargs='*')

    args = parser.parse_args()

    val_data = VapourData(args.data, mode='val', transforms=transforms_val, split=args.split is not None, split_factor=args.split)
    val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle=False, pin_memory=True)

    if args.oidx is not None:
        out_indices = [int(i) for i in args.oidx]
    else:
        out_indices = None

    model = UnetResEncoder(in_ch=3, out_ch=3, encoder_name='resnet34d', out_indices=out_indices).to(DEVICE)

    if args.checkpoint is not None:
        load_checkpoint(args.checkpoint, model)

    loss = torch.nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)

    metrics = val_fn(val_dataloader, model, loss, DEVICE, True)
    print(metrics)
    inpath = pathlib.Path(args.checkpoint)
    with open(os.path.join(args.outpath, (inpath.parent.name if args.modelname is None else args.modelname) + '.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['model name', 'loss', 'iou', 'recall', 'precision', 'mean radius error (correct droplets)', 'mean radius error (all droplets)'])
        writer.writerow([inpath.parent.name if args.modelname is None else args.modelname, metrics[0], metrics[1].item(), metrics[2], metrics[3], metrics[4], metrics[5]])


if __name__ == '__main__':
    main()
