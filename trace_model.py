import torch
from torch import jit
import argparse
from src.model import UnetResEncoder
from utils import load_checkpoint
import os

SUPPORTED_ARCH = ['unet']


def main():
    parser = argparse.ArgumentParser("Pickle a model so it can be used in the application later without referencing "
                                     "the architecture")

    parser.add_argument('--arch',
                        help=f'The architecture of the model. This script supports '
                             f'{"".join([(", " if i != 0 else "") + arch for i, arch in enumerate(SUPPORTED_ARCH)])}.',
                        default='unet')
    parser.add_argument('--inpath', help='The path to a checkpoint file to load.')
    parser.add_argument('--outpath', help='The path to save the model to.', default='.')

    args = parser.parse_args()

    if args.arch == 'unet':
        model = UnetResEncoder(out_ch=3)
    else:
        exit()

    if args.inpath is not None:
        load_checkpoint(args.inpath, model)

    outpath = os.path.join(args.outpath, f'{args.arch}.pt')

    x = torch.rand((3, 3, 224, 224))
    traced = jit.trace(model, x)
    print(traced.code)
    jit.save(traced, outpath)


if __name__ == '__main__':
    main()
