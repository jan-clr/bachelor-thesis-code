from detection import load_image
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        "This is a command line tool that uses a pytorch model to detect droplets in images of vapour in order to "
        "Measure their size.")
    parser.add_argument('--impath', help="The path to the directory where the image files for detection are placed",
                        metavar='IP', default='.')
    parser.add_argument('--model', help="Provide the path to a traced model to use instead of the built in one",
                        metavar='MP')
    parser.add_argument('--norm', help='Normalize the images before applying the model. '
                                       '\nThis may use drive space equivalent to the size of your data.',
                        action='store_true')
    return parser.parse_args()


def prepare_model(path, device):
    return torch.jit.load(path).to(device)
