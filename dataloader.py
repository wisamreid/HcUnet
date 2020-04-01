from __future__ import print_function, division
import os
from skimage import io
from torch.utils.data import Dataset
import glob as glob
import numpy as np


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class stack(Dataset):

    def __init__(self, path, mask_transforms, pwl_trasnforms, image_transforms, joint_transforms):
        """
        CSV File has a list of locations to other minibatch

        :param csv_file:
        :param transfrom:
        """

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.joint_transforms = joint_transforms
        self.pwl_transforms = pwl_trasnforms

        self.files = glob.glob(f'{path}{os.sep}*.mask.tif')

        if len(self.files) == 0:
            raise FileExistsError('No Valid Mask Files Found')



    def __len__(self):

        return len(self.files)

    def __getitem__(self, item):

        # Expect files to contain two endings *.mask.tif
        # with first run of os.path.splitext we remove .tif
        # with second pass we remove .mask
        file_with_mask = os.path.splitext(self.files[item])[0]

        image_data_path = os.path.splitext(file_with_mask)[0] + '.tif'
        pwl_data_path = os.path.splitext(file_with_mask)[0] + '.pwl.tif'
        mask_path = self.files[item]

        image = io.imread(image_data_path)
        mask = io.imread(mask_path)
        pwl = io.imread(pwl_data_path)

        # We have to assume there is always a channel index at the last dim
        # So for 3D its [Z,Y,X,C]
        # 2D: [Y,X,C]
        mask = np.expand_dims(mask, axis=mask.ndim)
        pwl = np.expand_dims(pwl, axis=pwl.ndim)

        # May Turn to Torch

        for mt in self.mask_transforms:
            mask = mt(mask)

        for pwlt in self.pwl_transforms:
            pwl = pwlt(pwl)

        for it in self.image_transforms:
            image = it(image)

        for jt in self.joint_transforms:
            image, mask, pwl = jt([image, mask, pwl])


        return image, mask, pwl

    @staticmethod
    def image(path, transform):

        image = io.imread(path)
        for t in transform:
            image = t(image)

        return image
