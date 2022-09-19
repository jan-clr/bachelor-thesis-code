import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.utils import extract_archive
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from cityscapesscripts.preparation import createTrainIdLabelImgs
import glob
import torchvision.transforms.functional as TF

from PIL import Image


class CustomCityscapesDataset(VisionDataset):
    """

    """
    # Based on https://github.com/mcordts/cityscapesScripts
    classes = Cityscapes.classes

    def __init__(self, root_dir: str = 'data', mode: str = 'train', id_to_use: str = 'labelTrainIds', transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, split: bool = False) -> None:

        super(CustomCityscapesDataset, self).__init__(root_dir, transforms, transform, target_transform)

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'leftImg8bit', mode)
        self.target_dir = os.path.join(root_dir, 'gtFine', mode)
        self.images = []
        self.targets = []
        self.split = split

        # Set env var for cityscapeScripts preparation
        os.environ['CITYSCAPES_DATASET'] = root_dir

        # Try to extract zips if unzipped files are not present
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.target_dir):
            image_dir_zip = os.path.join(self.root_dir, 'leftImg8bit_trainvaltest.zip')
            target_dir_zip = os.path.join(self.root_dir, 'gtFine_trainvaltest.zip')

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root_dir)
                extract_archive(from_path=target_dir_zip, to_path=self.root_dir)
            else:
                raise RuntimeError(
                    f"Dataset at '{root_dir}' not found or incomplete. Please make sure all required folders for the"
                    ' specified "mode" are inside the "root" directory')

        # generate label Ids for training
        if id_to_use == 'labelTrainIds':
            self.classes = list(filter(lambda cs_class: cs_class.train_id not in [-1, 255], Cityscapes.classes))
            if not glob.glob(f"{self.root_dir}/*/*/*/*labelTrainIds*"):
                createTrainIdLabelImgs.main()

        target_file_ending = f'gtFine_{id_to_use}.png'

        for city in os.listdir(self.image_dir):
            img_dir = os.path.join(self.image_dir, city)
            target_dir = os.path.join(self.target_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], target_file_ending
                )
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))

    def __len__(self) -> int:
        return len(self.images) * 4 if self.split else len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: The Index of the sample
        :return: (image, target) where target is the pixel level segmentation (labelIds) of the image
        """
        idx, splt_idx = divmod(index, 4)

        image = Image.open(self.images[index if not self.split else idx]).convert('RGB')
        target = Image.open(self.targets[index if not self.split else idx])

        if self.split:
            img_splt = TF.five_crop(image, (image.size[0] // 4, image.size[1] // 4))
            trg_splt = TF.five_crop(target, (target.size[0] // 4, target.size[1] // 4))
            image, target = img_splt[splt_idx], trg_splt[splt_idx]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target
