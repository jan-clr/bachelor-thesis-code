from src.utils import generate_pseudo_labels
from src.datasets import CustomCityscapesDataset
from src.transforms import transforms_generator
from src.model import UnetResEncoder

from torch.utils.data import DataLoader
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    dataset = CustomCityscapesDataset(root_dir='data/Cityscapes', transforms=transforms_generator, use_labeled=slice(0, 0),
                                  use_unlabeled=slice(0, None))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model = UnetResEncoder(in_ch=3, out_ch=3, encoder_name='resnet34d').to(DEVICE)
    generate_pseudo_labels(model, loader, 'data/test', device=DEVICE)


if __name__ == '__main__':
    main()