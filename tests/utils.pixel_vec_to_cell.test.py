from hcat.r_unet import RecursiveUnet as RUnet
import matplotlib.pyplot as plt
from hcat.unet import Unet_Constructor as Unet
from hcat import mask, utils, rcnn, transforms as t, segment
from hcat.loss import dice, cross_entropy
import hcat
import hcat.dataloader as dataloader
import torch.nn.functional as F
import torch
import sys



def does_it_crash():
    data = dataloader.RecursiveStack(path='../Data/train',
                                            joint_transforms=[t.to_float(),
                                                              t.reshape(),
                                                              ],
                                            image_transforms=[
                                                              t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                                              ]
                                            )

    _, mask, _, _, vector = data[0]

    labels = segment.pixel_vec_to_cell(vector, mask)

    return True

