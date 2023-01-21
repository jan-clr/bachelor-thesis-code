import torch
from app.procedures import prepare_model, parse_arguments
from importlib.resources import files

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
