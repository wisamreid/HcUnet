from __future__ import print_function, division
import os
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import glob as glob
import numpy as np
import ray
from PIL import Image, TiffImagePlugin

try:
    import transforms as t
except ModuleNotFoundError:
    import HcUnet.transforms as t


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class stack(Dataset):
    """
    Dataloader for unet

    """

    def __init__(self, path, image_transforms, joint_transforms, out_transforms=[t.to_tensor()]):
        """
        CSV File has a list of locations to other minibatch

        :param csv_file:
        :param transfrom:
        """


        self.image_transforms = image_transforms
        self.out_transforms = out_transforms
        self.joint_transforms = joint_transforms

        self.files = glob.glob(f'{path}{os.sep}*.mask.tif')

        if len(self.files) == 0:
            raise FileExistsError('No Valid Mask Files Found')

        self.image=[]
        self.mask=[]
        self.pwl=[]

        for file in self.files:
            file_with_mask = os.path.splitext(file)[0]

            image_data_path = os.path.splitext(file_with_mask)[0] + '.tif'
            pwl_data_path = os.path.splitext(file_with_mask)[0] + '.pwl.tif'
            mask_path = file
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


def test_image(path, transforms):
    image = io.imread(path)
    print(image.dtype)
    for t in transforms:
        print(t)
        image = t(image)
    return image[0]


class section(Dataset):
    """
    Dataloader for 2d faster rcnn


    """

    def __init__(self, path, image_transforms, joint_transforms, out_transforms=[t.to_tensor()]):
        """
        CSV File has a list of locations to other minibatch

        :param csv_file:
        :param transfrom:
        """

        self.image_transforms = image_transforms
        self.joint_transforms = joint_transforms
        self.out_transforms = out_transforms

        self.files = glob.glob(f'{path}{os.sep}*.xml')

        if len(self.files) == 0:
            raise FileExistsError('No COCO formated .xml files found')



    def __len__(self):

        return len(self.files)

    def __getitem__(self, item):

        image_data_path = os.path.splitext(self.files[item])[0] + '.tif'
        bbox_data_path = self.files[item]

        image = io.imread(image_data_path)

        tree = ET.parse(bbox_data_path)
        root = tree.getroot()

        bbox_loc = []
        classlabels = []

        for c in root.iter('object'):
            for cls in c.iter('name'):

                classlabels.append(cls.text)

            for a in c.iter('bndbox'):
                x1 = int(a[0].text)
                y1 = int(a[1].text)
                x2 = int(a[2].text)
                y2 = int(a[3].text)
                bbox_loc.append([x1, y1, x2, y2])

        for i,s in enumerate(classlabels):
            if s == 'OHC1':
                classlabels[i] = 1
            elif s == 'OHC2':
                classlabels[i] = 2
            elif s == 'OHC3':
                classlabels[i] = 3
            elif s == 'IHC':
                classlabels[i] = 4
            else:
                print(classlabels)
                print(bbox_loc)
                raise ValueError('Unidentified Label in XML file of ' + bbox_data_path)

        for it in self.image_transforms:
            image = it(image)

        for jt in self.joint_transforms:
            image, bbox_loc = jt(image, bbox_loc)

        for ot in self.out_transforms:
            image = ot(image)

        return image, {'boxes': torch.tensor(bbox_loc), 'labels':torch.tensor(classlabels)}

    @staticmethod
    def image(path, transform):

        image = io.imread(path)
        for t in transform:
            image = t(image)

        return image
