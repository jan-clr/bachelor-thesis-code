import torch
from app.procedures import prepare_model, parse_arguments, preprocess_imgs, process_imgs
from importlib.resources import files
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    The main entrypoint to the application.
    :return: nothing
    """
    args = parse_arguments()
    print(args)

    model_path = args.model if args.model is not None else files('app.data').joinpath('unet.pt')

    print(model_path)
    model = prepare_model(model_path, DEVICE)
    print(model.code)
    working_dir = args.impath

    if args.norm:
        working_dir = preprocess_imgs(working_dir)

    mask_dir = os.path.join(args.impath, 'masks')
    process_imgs(model, working_dir, mask_dir, bsize=10, device=DEVICE)