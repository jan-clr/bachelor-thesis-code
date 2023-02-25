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
import glob
from pprint import pprint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser("Validate a model for reporting.")

    parser.add_argument('--checkpoint', help='The path to a checkpoint file to load.', required=True)
    parser.add_argument('--pattern', help='If this is set to true, the checkpoint option will be treated as a glob '
                                          'style pattern and evaluation will be run on all matching files.'
                                          'This only works if the architecture of all models is the same.',
                        action='store_true')
    parser.add_argument('--filename', help='The name for the file the validation should be saved under.')
    parser.add_argument('--data', help='The path to a the validation dataset.', default='data/vapourbase')
    parser.add_argument('--outpath', help='The path to save the file to.', default='.')
    parser.add_argument('--split', help='The split factor of the images.', default=None, type=int)
    parser.add_argument('--bs', help='Batch size to use.', default=1, type=int)
    parser.add_argument('--oidx', help='Set the out indices for the feature extractor.', nargs='*')
    parser.add_argument('--append', help='Append the evaluation to an existing csv file instead of creating a new one.', action='store_true')
    parser.add_argument('--exclude', help='Exclude the given patter from the evaluation. Only relevant if --pattern is true.', default='')

    args = parser.parse_args()

    val_data = VapourData(args.data, mode='test', transforms=transforms_val, split=args.split is not None, split_factor=args.split)
    val_dataloader = DataLoader(val_data, batch_size=args.bs, shuffle=False, pin_memory=True)

    if args.oidx is not None:
        out_indices = [int(i) for i in args.oidx]
    else:
        out_indices = None

    model = UnetResEncoder(in_ch=3, out_ch=3, encoder_name='resnet34d', out_indices=out_indices).to(DEVICE)

    checkpoints = [args.checkpoint] if not args.pattern else [p for p in (set(glob.glob(args.checkpoint, recursive=True)) - set(glob.glob(args.exclude, recursive=True)))]
    pprint(checkpoints)
    print(len(checkpoints))
    append = args.append
    for checkpoint in checkpoints:
        if args.pattern:
            print(f'\nValidating {checkpoint}')
        load_checkpoint(checkpoint, model)

        loss = torch.nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)

        metrics = val_fn(val_dataloader, model, loss, DEVICE, True)
        pprint(metrics)
        inpath = pathlib.Path(checkpoint)
        with open(os.path.join(args.outpath, (inpath.parent.name if args.filename is None else args.filename) + '.csv'), 'w' if not append else 'a') as csv_file:
            writer = csv.writer(csv_file)
            if not append:
                writer.writerow(['model name', 'loss', 'iou', 'recall', 'precision', 'mean radius error (correct droplets)', 'mean radius error (all droplets)'])
            writer.writerow([inpath.parent.name, metrics[0], metrics[1].item(), metrics[2], metrics[3], metrics[4], metrics[5]])
        # always append after the first iteration
        append = True


if __name__ == '__main__':
    main()
