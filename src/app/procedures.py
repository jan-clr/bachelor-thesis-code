import glob

import cv2

from src.detection import load_image
import torch
import argparse
from src.normalize import ImgLoader, normalize_images_batched
import os
from tqdm import tqdm
from pathlib import Path
import shutil
from src.transforms import transform_eval
from torchvision.utils import save_image

ID_TO_COLOR = {
    0: (0, 0, 0),
    1: (51, 221, 255),
    2: (250, 125, 187),
    3: (245, 147, 49),
    4: (140, 120, 240)
}


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
    parser.add_argument('--device', help="Set device manually.")
    return parser.parse_args()


def prepare_model(path, device):
    return torch.jit.load(path).to(device).eval()


def unfold_sliding_window(img):
    if len(img.shape) < 4:
        img = img.unsqueeze(0)
    torch.nn.functional.unfold(img, kernel_size=(224, 224))
    pass


def transform_inputs(images):
    tensors = []
    for img in images:
        tensor = transform_eval(img)
        tensors.append(tensor)
    # TODO sliding window construct

    return torch.stack(tensors)


def mask_to_col(mask):
    col_mask = torch.zeros((*mask.shape, 3))
    for id in torch.unique(mask):
        col_mask[mask == id] = torch.Tensor(
            ID_TO_COLOR[id.item()] if id.item() in ID_TO_COLOR.keys() else (255, 255, 255))

    return col_mask.permute(2, 0, 1)


def create_overlays(impath, maskpath, outpath):
    images, masks = [], []
    img_files = sorted(glob.glob(os.path.join(impath, '*.*')))
    mask_files = sorted(glob.glob(os.path.join(maskpath, '*color.*')))
    print(img_files)
    print(mask_files)
    for img_file, mask_file in zip(img_files, mask_files):
        img_path = Path(img_file)
        img = cv2.imread(img_file)
        mask = cv2.imread(mask_file)
        overlay = cv2.addWeighted(mask, 0.45, img, 1.0, 0)
        cv2.imwrite(os.path.join(outpath, f"{img_path.stem}_overlay.png"), overlay)


def preprocess_imgs(impath, bsize=10):
    norm_path = os.path.join(impath, 'normalized')
    normalize_images_batched(inpath=impath, outpath=norm_path, bsize=bsize)
    return norm_path


def process_imgs(model, impath, outpath, bsize=1, device='cpu'):
    path = Path(outpath)
    if path.exists():
        delete = input('Folder already exists. Delete? [y/n]:')
        if delete == 'y':
            shutil.rmtree(path)
        else:
            exit()
    path.mkdir(parents=True, exist_ok=False)

    loader = ImgLoader(impath, batch_size=int(bsize))
    loop = tqdm(enumerate(loader), total=len(loader))
    with torch.no_grad():
        for batch, (images, files) in loop:
            images = transform_inputs(images).to(device)
            predictions = model(images)
            # TODO sliding window reconstruct
            masks = torch.argmax(predictions, dim=1)
            for i, mask in enumerate(masks):
                img_path = Path(files[i])
                save_image(mask / 255.0, os.path.join(outpath, f"{img_path.stem}_ids.png"))
                save_image(mask_to_col(mask) / 255.0, os.path.join(outpath, f"{img_path.stem}_color.png"))


def postprocess_masks():
    pass


def main():
    create_overlays(impath='../../data/vdetect_test', maskpath='../../data/vdetect_test/masks', outpath='../../data/vdetect_test/masks')


if __name__ == '__main__':
    main()
