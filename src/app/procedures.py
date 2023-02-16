import glob

import cv2
import numpy as np

from src.detection import load_image, detect_droplets, Droplet, detect_streaks, draw_droplets, draw_streaks
import torch
import argparse
from src.normalize import ImgLoader, normalize_images_batched
import os
from tqdm import tqdm
from pathlib import Path
from src.transforms import transform_eval
from src.utils import create_dir
from torchvision.utils import save_image
import csv

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
    parser.add_argument('--labelpath', help='Start from existing labels.')
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
    img_files = sorted(glob.glob(os.path.join(impath, '*.*')))
    mask_files = sorted(glob.glob(os.path.join(maskpath, '*color.*')))
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
    create_dir(outpath)

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


def measure_droplets(maskpath):
    mask_files = sorted(glob.glob(os.path.join(maskpath, '*ids.*')))
    droplets = {}
    streaks = {}
    for mask_file in mask_files:
        path = Path(mask_file)
        mask = load_image(mask_file)
        print(mask_file)
        dd = detect_droplets(mask)
        #ds = detect_streaks(mask)
        droplets[f"{path.stem[:-len('_ids')]}.png"] = dd
        #streaks[f"{path.stem[:-len('_ids')]}.png"] = ds

    return droplets, streaks


def draw_detected_objects(impath, outpath, droplets: dict[str, [Droplet]], streaks):
    img_files = sorted(glob.glob(os.path.join(impath, '*.*')))
    for filepath in img_files:
        img = None
        path = Path(filepath)
        filename = path.name
        if droplets is not None and filename in droplets.keys():
            img = load_image(filepath, False)
            img = draw_droplets(img, droplets[filename])
        if streaks is not None and filename in streaks.keys():
            if img is None:
                img = load_image(filepath, False)
            img = draw_streaks(img, streaks[filename])
        if img is not None:
            cv2.imwrite(os.path.join(outpath, f"{path.stem}_detected.png"), img)


def droplets_to_csv(droplets: dict[str, [Droplet]], outpath):
    flat_rad = np.array([droplet.radius for dlist in droplets.values() for droplet in dlist])
    rad_mean = np.mean(flat_rad)
    rad_std = np.std(flat_rad)
    with open(outpath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['radius mean', rad_mean])
        writer.writerow(['radius std', rad_std])
        writer.writerow([])
        writer.writerow(['filename', 'center_x', 'center_y', 'radius'])
        for filename, droplet_in_file in droplets.items():
            writer.writerow([filename])
            for droplet in droplet_in_file:
                writer.writerow(['', droplet.center[1], droplet.center[0], droplet.radius])
    return rad_mean, rad_std, flat_rad


def streaks_to_csv(streaks, outpath):
    pass


def main():
    droplets, streaks = measure_droplets('../../data/vdetect_test/masks')
    #draw_detected_objects('../../data/vdetect_test', '../../data/vdetect_test/masks', droplets, None)
    droplets_to_csv(droplets, '../../data/vdetect_test/masks/droplets.csv')


if __name__ == '__main__':
    main()
