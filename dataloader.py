from __future__ import print_function, division
import os
from skimage import io
from torch.utils.data import Dataset
import glob as glob
import numpy as np
from PIL import Image, TiffImagePlugin
try:
    import transforms as t
except ModuleNotFoundError:
    import HcUnet.transforms as t


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class stack(Dataset):

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
