import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.utils import extract_archive
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from cityscapesscripts.preparation import createTrainIdLabelImgs
import glob
import torchvision.transforms.functional as TF
from PIL import Image
from utils import resize_images, split_images
import cv2


class CustomCityscapesDataset(VisionDataset):
    """

    """
    # Based on https://github.com/mcordts/cityscapesScripts
    classes = Cityscapes.classes

    def __init__(self, root_dir: str = 'data', mode: str = 'train', id_to_use: str = 'labelTrainIds', transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, low_res: bool = True) -> None:

        super(CustomCityscapesDataset, self).__init__(root_dir, transforms, transform, target_transform)

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, f'leftImg8bit{"_lowres" if low_res else ""}', mode)
        self.target_dir = os.path.join(root_dir, f'gtFine{"_lowres" if low_res else ""}', mode)
        self.images = []
        self.targets = []

        # Set env var for cityscapeScripts preparation
        os.environ['CITYSCAPES_DATASET'] = root_dir

        print(self.image_dir, self.target_dir)
        # look for full res files or zips if dirs not present
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.target_dir):
            # when using downsampled images, check if full res images available
            imdir_full, tardir_full = os.path.join(root_dir, 'leftImg8bit', mode), os.path.join(root_dir, 'gtFine', mode)
            if low_res and os.path.isdir(imdir_full) and os.path.isdir(tardir_full):
                resize_images(from_path=imdir_full, to_path=self.image_dir, size=(256, 512))
                resize_images(from_path=tardir_full, to_path=self.target_dir, size=(256, 512), anti_aliasing=False)
            else:
                # Try to extract zips if unzipped files are not present
                image_dir_zip = os.path.join(self.root_dir, 'leftImg8bit_trainvaltest.zip')
                target_dir_zip = os.path.join(self.root_dir, 'gtFine_trainvaltest.zip')

                if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                    extract_archive(from_path=image_dir_zip, to_path=self.root_dir)
                    extract_archive(from_path=target_dir_zip, to_path=self.root_dir)
                    # generate label Ids for training
                    if id_to_use == 'labelTrainIds':
                        if not glob.glob(f"{self.root_dir}/*/*/*/*labelTrainIds*"):
                            createTrainIdLabelImgs.main()
                    if low_res:
                        resize_images(from_path=imdir_full, to_path=self.image_dir, size=(256, 512))
                        resize_images(from_path=tardir_full, to_path=self.target_dir, size=(256, 512), anti_aliasing=False)
                else:
                    raise RuntimeError(
                        f"Dataset at '{root_dir}' not found or incomplete. Please make sure all required folders for the"
                        ' specified "mode" are inside the "root" directory')

        if id_to_use == 'labelTrainIds':
            self.classes = list(filter(lambda cs_class: cs_class.train_id not in [-1, 255], Cityscapes.classes))

        target_file_ending = f'gtFine_{id_to_use}.png'

        # add files to index
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
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: The Index of the sample
        :return: (image, target) where target is the pixel level segmentation (labelIds) of the image
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target


class VapourData(VisionDataset):
    """

    """
    classes = [{'id': 0, 'name': 'background'}, {'id': 1, 'name': 'droplet_border'}, {'id': 2, 'name': 'droplet_inside'}]

    def __init__(self, root_dir: str = 'data', mode: str = 'train', id_to_use: str = 'labelIds', transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, low_res: bool = False, split: bool = True) -> None:

        super(VapourData, self).__init__(root_dir, transforms, transform, target_transform)

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, f'leftImg8bit{"_lowres" if low_res else ""}{"_split" if split else ""}', mode)
        self.target_dir = os.path.join(root_dir, f'gtFine{"_lowres" if low_res else ""}{"_split" if split else ""}', mode)
        self.images = []
        self.targets = []

        tl_imdir, tl_tardir = 'leftImg8bit', 'gtFine'
        print(f"Loading Data from:\nImage Dir: {self.image_dir}\nTarget Dir: {self.target_dir}")
        # look for full res files or zips if dirs not present
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.target_dir):
            print("Data not found with specified parameters. Trying to reconstruct.")

            # when using downsampled images, check if full res images available
            self.image_dir, self.target_dir = os.path.join(root_dir, tl_imdir, mode), os.path.join(root_dir, tl_tardir, mode)

            # Try to extract zips if unzipped files are not present
            image_dir_zip = os.path.join(self.root_dir, 'leftImg8bit.zip')
            target_dir_zip = os.path.join(self.root_dir, 'gtFine.zip')

            if not (os.path.isdir(self.image_dir) and os.path.isdir(self.target_dir)):
                if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                    print("Full scale image files not found. Extracting from archive.")
                    extract_archive(from_path=image_dir_zip, to_path=self.root_dir)
                    extract_archive(from_path=target_dir_zip, to_path=self.root_dir)
                else:
                    raise RuntimeError(
                        f"Dataset at '{root_dir}' not found or incomplete. Please make sure all required folders for the"
                        ' specified "mode" are inside the "root" directory')

            # Update top level img dir in case lowres is used
            if low_res:
                tl_imdir, tl_tardir = f'{tl_imdir}_lowres', f'{tl_tardir}_lowres'
                imdir_to = os.path.join(root_dir, tl_imdir, mode)
                tardir_to = os.path.join(root_dir, tl_tardir, mode)

            # resize if necessary
            if low_res and os.path.isdir(self.image_dir) and os.path.isdir(self.target_dir) and not (os.path.isdir(imdir_to) and os.path.isdir(tardir_to)):
                resize_images(from_path=self.image_dir, to_path=imdir_to, size=(256, 512))
                resize_images(from_path=self.target_dir, to_path=tardir_to, size=(256, 512), anti_aliasing=False)

                self.image_dir, self.target_dir = imdir_to, tardir_to

            # Update top level img dir in case split is used
            if split:
                tl_imdir, tl_tardir = f'{tl_imdir}_split', f'{tl_tardir}_split'
                imdir_to = os.path.join(root_dir, tl_imdir, mode)
                tardir_to = os.path.join(root_dir, tl_tardir, mode)

            # Save crops if necessary
            if split and os.path.isdir(self.image_dir) and os.path.isdir(self.target_dir) and not (os.path.isdir(imdir_to) and os.path.isdir(tardir_to)):
                split_images(from_path=self.image_dir, to_path=imdir_to, file_ext='tif')
                split_images(from_path=self.target_dir, to_path=tardir_to)

                self.image_dir, self.target_dir = imdir_to, tardir_to

        target_file_ending = f'gtFine_{id_to_use}.png'

        # add files to index
        for file_name in os.listdir(self.image_dir):
            target_name = "{}_{}".format(
                file_name.split("_leftImg8bit")[0], target_file_ending
            )
            self.images.append(os.path.join(self.image_dir, file_name))
            self.targets.append(os.path.join(self.target_dir, target_name))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: The Index of the sample
        :return: (image, target) where target is the pixel level segmentation (labelIds) of the image
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target


def main():
    train_data = VapourData("../data/vapourbase", mode='train', split=True)


if __name__ == '__main__':
    main()
