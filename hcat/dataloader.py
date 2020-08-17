from __future__ import print_function, division
import os
import torch
from skimage import io
from torch.utils.data import Dataset
import xml.etree.ElementTree
import glob as glob
import numpy as np
import hcat.transforms as t

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Stack(Dataset):
    """
    Dataloader for hcat.unet
    Processes 3D images

    Performs
    """
    def __init__(self, path, image_transforms, joint_transforms, out_transforms=None):
        """

        :param path: str
        :param image_transforms: list
        :param joint_transforms: list
        :param out_transforms: list
        """

        if out_transforms is None:
            out_transforms = [t.to_tensor()]

        self.image_transforms = image_transforms
        self.out_transforms = out_transforms
        self.joint_transforms = joint_transforms

        self.files = glob.glob(f'{path}{os.sep}*.mask.tif')

        if len(self.files) == 0:
            raise FileExistsError('No Valid Mask Files Found')

        self.image = []
        self.mask = []
        self.pwl = []

        for mask_path in self.files:
            file_with_mask = os.path.splitext(mask_path)[0]
            image_data_path = os.path.splitext(file_with_mask)[0] + '.tif'
            pwl_data_path = os.path.splitext(file_with_mask)[0] + '.pwl.tif'
            self.image.append(io.imread(image_data_path))
            try:
                self.mask.append(io.imread(mask_path)[:, :, :, 0])
            except IndexError:
                self.mask.append(io.imread(mask_path))

            self.pwl.append(io.imread(pwl_data_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):

        # Expect files to contain two endings *.mask.tif
        # with first run of os.path.splitext we remove .tif
        # with second pass we remove .mask

        image = self.image[item]
        mask = self.mask[item]
        pwl = self.pwl[item]

        # We have to assume there is always a channel index at the last dim
        # So for 3D its [Z,Y,X,C]
        # 2D: [Y,X,C]
        mask = np.expand_dims(mask, axis=mask.ndim)
        pwl = np.expand_dims(pwl, axis=pwl.ndim)

        # May Turn to Torch
        for jt in self.joint_transforms:
            image, mask, pwl = jt([image, mask, pwl])
        for it in self.image_transforms:
            image = it(image)
        for ot in self.out_transforms:
            image, mask, pwl = ot([image, mask, pwl])

        return image, mask, pwl


class Section(Dataset):
    """
    Dataloader for 2d faster rcnn
    """

    def __init__(self, path, image_transforms, joint_transforms, out_transforms=None, simple_class=False):
        """

        :param path:
        :param image_transforms:
        :param joint_transforms:
        :param out_transforms:
        """


        if out_transforms is None:
            out_transforms = [t.to_tensor()]

        self.image_transforms = image_transforms
        self.joint_transforms = joint_transforms
        self.out_transforms = out_transforms

        # If true, reduces ever OHC1, OHC2, OHC3 to one class of just OHC's
        self.simple_class = simple_class

        self.files = glob.glob(f'{path}{os.sep}*.xml')

        if len(self.files) == 0:
            raise FileNotFoundError(f'No COCO formatted xml files found in {self.files}')

    def __len__(self):
        """
        :return:
        """
        return len(self.files)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """

        image_data_path = os.path.splitext(self.files[item])[0] + '.tif'
        bbox_data_path = self.files[item]
        image = io.imread(image_data_path)
        tree = xml.etree.ElementTree.parse(bbox_data_path)
        root = tree.getroot()

        bbox_loc = []
        class_labels = []

        for c in root.iter('object'):
            for cls in c.iter('name'):

                class_labels.append(cls.text)

            for a in c.iter('bndbox'):
                x1 = int(a[0].text)
                y1 = int(a[1].text)
                x2 = int(a[2].text)
                y2 = int(a[3].text)
                bbox_loc.append([x1, y1, x2, y2])

        for i, s in enumerate(class_labels):
            if s == 'OHC1':
                class_labels[i] = 1
            elif s == 'OHC2':
                class_labels[i] = 2
            elif s == 'OHC3':
                class_labels[i] = 3
            elif s == 'IHC':
                class_labels[i] = 4
            else:
                print(class_labels)
                print(bbox_loc)
                raise ValueError('Unidentified Label in XML file of ' + bbox_data_path)

        class_labels = torch.tensor(class_labels)

        # For testing - reduce row specific labels in cell specific (IHC or OHC)
        if self.simple_class:
            class_labels[class_labels == 2] = 1  # OHC2 -> OHC
            class_labels[class_labels == 3] = 1  # OHC3 -> OHC
            class_labels[class_labels == 4] = 2  # IHC

        for it in self.image_transforms:
            image = it(image)
        for jt in self.joint_transforms:
            image, bbox_loc = jt(image, bbox_loc)
        for ot in self.out_transforms:
            image = ot(image)

        return image, {'boxes': torch.tensor(bbox_loc), 'labels': torch.tensor(class_labels)}
