import torch
import skimage.transform as transform
import scipy.ndimage as ndimage
import skimage.exposure as exposure
import numpy as np
import matplotlib.pyplot as plt


def int16_to_float(image: np.int16):
    if not image.dtype == 'uint16':
        raise TypeError(f'Expected image datatype to be uint16 but got {image.dtype}')
    return (image/2**16).astype(np.float)


def int8_to_float(image: np.int) -> np.ndarray:
    if not image.dtype == 'uint8':
        raise TypeError(f'Expected image datatype to be uint8 but got {image.dtype}')

    return(image/255).astype(np.float)

def to_tensor(image_list: list) -> torch.tensor:
    if not isinstance(image_list, list):
        raise TypeError(f'Expected list but got {type(image_list)}')
    out = []
    for image in image_list:
        num_dims = len(image.shape)
        image = torch.tensor(image)
        out.append(image.unsqueeze(0).transpose(num_dims, 0).squeeze(dim=image.dim()).unsqueeze(0))

    # [x,y,z,c] -> [1,x,y,z,c] -> [c,x,y,z,1] -> [c,x,y,z] -> [1,c,x,y,z]
    return out

def reshape(image_list: list):
    """
    Expects image dimmensions to be [Z,Y,X,C] or [Y,X,C]
        (this is how skimage.io.imread outputs 3D tifs), we add a channel if necessary

    Reshapes to [x,y,z,c]

    :param image:
    :return:
    """
    if not isinstance(image_list, list):
        raise TypeError(f'Expected input type of list but got {type(image_list)}')

    out = []

    for image in image_list:
        out.append(image.swapaxes(len(image.shape)-2, 0))

    return out


def spekle(image):

    if not image.dtype == 'float':
        raise TypeError(f'Expected image datatype to be float but got {image.dtype}')

    image_shape = np.shape(image)
    noise = np.random.normal(0,.1,image_shape)
    noise = np.float32(noise)
    image = image+noise
    image[image<0] = 0
    image[image>1] = 1

    return image

def random_gamma(image: np.float):
    if not image.dtype == 'float':
        raise TypeError(f'Expected image dataype to be float but got {image.dtype}')

    factor = np.random.uniform(.5, 1.5, 1)
    if factor < 0:
        factor = 0

    return exposure.adjust_gamma(image, factor)

def random_rotate(image_list: list):
    """
    Expects a list of numpy.ndarrays of all the same shape. Randonmly rotates the image along x or y dimmension
    and returns list of rotated images

    :param image:
    :return:
    """

    if not isinstance(image_list, list ):
        raise TypeError(f'Expected list of images as input, but got {type(image_list)}')

    theta = np.random.randint(0, 360, 1)[0]
    out = []
    for image in image_list:
        rot_image = ndimage.rotate(image, angle=theta, reshape='true')
        out.append(rot_image)

    return out

