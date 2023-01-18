import os

import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.utils import extract_archive
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from cityscapesscripts.preparation import createTrainIdLabelImgs
import glob
import torchvision.transforms.functional as TF
from PIL import Image
from src.utils import resize_images, split_images
import cv2
import torch.multiprocessing


torch.multiprocessing.set_sharing_strategy('file_system')

NO_LABEL = 255


class CustomCityscapesDataset(VisionDataset):
    """

    """
    # Based on https://github.com/mcordts/cityscapesScripts
    classes = Cityscapes.classes

    def __init__(self, root_dir: str = 'data', mode: str = 'train', id_to_use: str = 'labelTrainIds',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None,
                 low_res: bool = True,
                 use_labeled: slice = None,
                 use_unlabeled: slice = None,
                 use_pseudo_labels: slice = None,
                 pseudo_label_dir: str = None) -> None:

        super(CustomCityscapesDataset, self).__init__(root_dir, transforms, transform, target_transform)

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, f'leftImg8bit{"_lowres" if low_res else ""}', mode)
        self.target_dir = os.path.join(root_dir, f'gtFine{"_lowres" if low_res else ""}', mode)
        self.images = []
        self.targets = []

        # Set env var for cityscapeScripts preparation
        os.environ['CITYSCAPES_DATASET'] = root_dir

        # look for full res files or zips if dirs not present
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.target_dir):
            # when using downsampled images, check if full res images available
            imdir_full, tardir_full = os.path.join(root_dir, 'leftImg8bit', mode), os.path.join(root_dir, 'gtFine',
                                                                                                mode)
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
                        resize_images(from_path=tardir_full, to_path=self.target_dir, size=(256, 512),
                                      anti_aliasing=False)
                else:
                    raise RuntimeError(
                        f"Dataset at '{root_dir}' not found or incomplete. Please make sure all required folders for the"
                        ' specified "mode" are inside the "root" directory')

        if id_to_use == 'labelTrainIds':
            self.classes = list(filter(lambda cs_class: cs_class.train_id not in [-1, 255], Cityscapes.classes))

        target_file_ending = f'gtFine_{id_to_use}.png'

        # add files to index
        for city in sorted(os.listdir(self.image_dir)):
            img_dir = os.path.join(self.image_dir, city)
            target_dir = os.path.join(self.target_dir, city)
            for file_name in sorted(os.listdir(img_dir)):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], target_file_ending
                )
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))

        # change targets for images in pseudo label range
        if use_pseudo_labels is not None and pseudo_label_dir is not None:
            print(use_pseudo_labels, pseudo_label_dir)
            pseudo_label_files = sorted(os.listdir(pseudo_label_dir))
            pseudo_targets = self.targets[use_pseudo_labels]
            for i in range(len(pseudo_targets)):
                pseudo_targets[i] = os.path.join(pseudo_label_dir, pseudo_label_files[i])
            self.targets[use_pseudo_labels] = pseudo_targets

        # default use all images with targets
        self.labeled_idxs = [idx for idx in range(len(self.images))]
        self.unlabeled_idxs = []

        # Update lists if arguments are set
        if use_labeled is not None:
            self.labeled_idxs = self.labeled_idxs[use_labeled]
        if use_unlabeled is not None:
            self.unlabeled_idxs = [idx for idx in range(len(self.images))][use_unlabeled]

        # make unlabeled slices take priority and ensure no overlap
        self.labeled_idxs = [idx for idx in self.labeled_idxs if idx not in self.unlabeled_idxs]

        # reorder images and targets so dataset indexing can always begin at 0
        images = [self.images[i] for i in self.labeled_idxs]
        images += [self.images[i] for i in self.unlabeled_idxs]
        self.images = images
        self.targets = [self.targets[i] for i in self.labeled_idxs]

        # Update idxs
        self.labeled_idxs = [i for i in range(len(self.labeled_idxs))]
        self.unlabeled_idxs = [i for i in range(len(self.labeled_idxs), len(self.labeled_idxs) + len(self.unlabeled_idxs))]

    def __len__(self) -> int:
        return len(self.labeled_idxs) + len(self.unlabeled_idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: The Index of the sample
        :return: (image, target) where target is the pixel level segmentation (labelIds) of the image
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = None
        if index in self.labeled_idxs:
            target = Image.open(self.targets[index]).convert('L')
        elif index in self.unlabeled_idxs:
            target = np.full((image.size[0], image.size[1]), NO_LABEL, dtype='uint8')
        else:
            raise IndexError("Index is neither in labeled nor in unlabeled subset.")

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
    classes = [{'id': 0, 'name': 'background'},
               {'id': 1, 'name': 'droplet_streak'},
               {'id': 2, 'name': 'droplet_border'},
               {'id': 3, 'name': 'droplet_inside'},
               {'id': 4, 'name': 'small_droplet'}]

    def __init__(self, root_dir: str = 'data', mode: str = 'train', id_to_use: str = 'labelIds',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None, low_res: bool = False, split: bool = True,
                 use_labeled: slice = None,
                 use_unlabeled: slice = None,
                 use_pseudo_labels: slice = None,
                 pseudo_label_dir: str = None,
                 ignore_labels=None) -> None:

        super(VapourData, self).__init__(root_dir, transforms, transform, target_transform)

        if ignore_labels is None:
            ignore_labels = [3, 4]
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, f'leftImg8bit{"_lowres" if low_res else ""}{"_split" if split else ""}',
                                      mode)
        self.target_dir = os.path.join(root_dir, f'gtFine{"_lowres" if low_res else ""}{"_split" if split else ""}',
                                       mode)
        self.images = []
        self.targets = []
        self.ignore = ignore_labels
        self.classes = [lab for lab in self.classes if lab['id'] not in ignore_labels]
        self.used_ids = [x['id'] for x in self.classes]

        tl_imdir, tl_tardir = 'leftImg8bit', 'gtFine'
        print(f"Loading Data from:\nImage Dir: {self.image_dir}\nTarget Dir: {self.target_dir}")
        # look for full res files or zips if dirs not present
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.target_dir):
            print("Data not found with specified parameters. Trying to reconstruct.")

            # when using downsampled images, check if full res images available
            self.image_dir, self.target_dir = os.path.join(root_dir, tl_imdir, mode), os.path.join(root_dir, tl_tardir,
                                                                                                   mode)

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
            if low_res and os.path.isdir(self.image_dir) and os.path.isdir(self.target_dir) and not (
                    os.path.isdir(imdir_to) and os.path.isdir(tardir_to)):
                resize_images(from_path=self.image_dir, to_path=imdir_to, size=(256, 512))
                resize_images(from_path=self.target_dir, to_path=tardir_to, size=(256, 512), anti_aliasing=False)

                self.image_dir, self.target_dir = imdir_to, tardir_to

            # Update top level img dir in case split is used
            if split:
                tl_imdir, tl_tardir = f'{tl_imdir}_split', f'{tl_tardir}_split'
                imdir_to = os.path.join(root_dir, tl_imdir, mode)
                tardir_to = os.path.join(root_dir, tl_tardir, mode)

            # Save crops if necessary
            if split and os.path.isdir(self.image_dir) and os.path.isdir(self.target_dir) and not (
                    os.path.isdir(imdir_to) and os.path.isdir(tardir_to)):
                split_images(from_path=self.image_dir, to_path=imdir_to, file_ext='*')
                split_images(from_path=self.target_dir, to_path=tardir_to)

                self.image_dir, self.target_dir = imdir_to, tardir_to

        target_file_ending = f'gtFine_{id_to_use}.png'

        # add files to index
        unlabeled_images = []
        for dir_name in sorted(os.listdir(self.image_dir)):
            tardir = os.path.join(self.target_dir, dir_name)
            imdir = os.path.join(self.image_dir, dir_name)
            for file_name in sorted(os.listdir(imdir)):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], target_file_ending
                )
                file_path, target_path = os.path.join(imdir, file_name), os.path.join(tardir, target_name)
                if os.path.isfile(target_path):
                    self.images.append(file_path)
                    self.targets.append(target_path)
                else:
                    unlabeled_images.append(file_path)

        print(f"Found {len(self.images)} labeled and {len(unlabeled_images)} unlabeled images.")

        # default use all images with targets
        self.labeled_idxs = [idx for idx in range(len(self.images))]
        self.unlabeled_idxs = []

        if use_pseudo_labels or use_unlabeled is not None:
            self.images += unlabeled_images

        # change targets for images in pseudo label range
        if use_pseudo_labels is not None and pseudo_label_dir is not None:
            pseudo_label_files = sorted(os.listdir(pseudo_label_dir))
            pseudo_targets = self.targets[use_pseudo_labels]
            for i in range(len(pseudo_targets)):
                pseudo_targets[i] = os.path.join(pseudo_label_dir, pseudo_label_files[i])
            self.targets[use_pseudo_labels] = pseudo_targets

        # Update lists if arguments are set
        if use_labeled is not None:
            self.labeled_idxs = self.labeled_idxs[use_labeled]
        if use_unlabeled is not None:
            self.unlabeled_idxs = [idx for idx in range(len(self.images))][use_unlabeled]

        # make unlabeled slices take priority and ensure no overlap
        self.labeled_idxs = [idx for idx in self.labeled_idxs if idx not in self.unlabeled_idxs]

        # reorder images and targets so dataset indexing can always begin at 0
        images = [self.images[i] for i in self.labeled_idxs]
        images += [self.images[i] for i in self.unlabeled_idxs]
        self.images = images
        self.targets = [self.targets[i] for i in self.labeled_idxs]

        # Update idxs
        self.labeled_idxs = [i for i in range(len(self.labeled_idxs))]
        self.unlabeled_idxs = [i for i in range(len(self.labeled_idxs), len(self.unlabeled_idxs))]
        print(f"Using {len(self.images)} samples.")

    def __len__(self) -> int:
        return len(self.labeled_idxs) + len(self.unlabeled_idxs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        :param index: The Index of the sample
        :return: (image, target) where target is the pixel level segmentation (labelIds) of the image
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = None
        if index in self.labeled_idxs:
            target = np.array(Image.open(self.targets[index]))
            # Remove ignored labels and make sure all remaining labels have consecutive ids
            for label_id in self.ignore:
                target[target == label_id] = 0
            for i, used_id in enumerate(self.used_ids):
                target[target == used_id] = i
        elif index in self.unlabeled_idxs:
            target = np.full((image.size[0], image.size[1]), NO_LABEL)
        else:
            raise IndexError("Index is neither in labeled nor in unlabeled subset.")

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target


def main():
    train_data = VapourData("../data/vapourbase", mode='train', split=True, use_labeled=slice(None, None), use_unlabeled=None)


if __name__ == '__main__':
    main()
