import torch
import scipy.ndimage as ndimage
import skimage.exposure as exposure
import numpy as np


# DECORATOR
def joint_transform(func):
    """
    This wrapper generates a seed value for each transform and passes an image from a list to that function
    It then passes images from a list to the function one at a time and returns a list of outputs

    :param func: Function with arguments 'image' and 'seed'
    :return: Wrapped function that can now accept lists
    """

    def wrapper(*args):
        image_list = args[-1]  # In the case of a class function, there may be two args, one is 'self'
        # We only want to take the last argument, which should always be the image list
        if not type(image_list) == list:
            image_list = [image_list]

        if len(image_list) > 1:
            for i in range(len(image_list) - 1):
                if not image_list[i].ndim == image_list[i + 1].ndim:
                    raise ValueError('Images in joint transforms do not contain identical dimensions.'
                                     + f'Im {i}.ndim:{image_list[i].ndim} != Im {i + 1}.ndim:{image_list[i + 1].ndim} ')
        out = []
        seed = np.random.randint(0, 1e8, 1)
        for im in image_list:
            if len(args) > 1:
                out.append(func(args[0], image=im, seed=seed))
            else:
                out.append(func(image=im, seed=seed))
        if len(out) == 1:
            out = out[0]
        return out

    return wrapper


class int16_to_float:
    def __init__(self):
        pass

    def __call__(self, image: np.int16):
        if not image.dtype == 'uint16':
            raise TypeError(f'Expected image datatype to be uint16 but got {image.dtype}')
        return (image / 2 ** 16).astype(np.float)


class int8_to_float:
    def __init__(self):
        pass

    def __call__(self, image: np.int) -> np.ndarray:
        if not image.dtype == 'uint8':
            raise TypeError(f'Expected image datatype to be uint8 but got {image.dtype}')
        return (image / 255).astype(np.float)


class to_float:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        if image.dtype == 'uint16':
            out = (image / 2 ** 16).astype(np.float)
        elif image.dtype == 'uint8':
            out = (image / 255).astype(np.float)
        elif image.dtype == 'float':
            out = image
        else:
            raise TypeError(f'Expected image datatype of uint8 or uint16 but got {image.dtype}')
        return out


class to_tensor:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        """
        Function which reformats a numpy array of [x,y,z,c] to [1, c, x, y, z]

        :param image: 2D or 3D ndarray with the channel in the last index
        :return: 2D or 3D torch.Tensor formated for input into a convolutional neural net
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list but got {type(image)}')
        num_dims = len(image.shape)
        image = torch.tensor(image)
        # Performs these operations in this order...
        # [x,y,z,c] -> [1,x,y,z,c] -> [c,x,y,z,1] -> [c,x,y,z] -> [1,c,x,y,z]
        return image.unsqueeze(0).transpose(num_dims, 0).squeeze(dim=image.dim()).unsqueeze(0)


class reshape:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        """
        Expects image dimmensions to be [Z,Y,X,C] or [Y,X,C]
            (this is how skimage.io.imread outputs 3D tifs), we add a channel if necessary

        Reshapes to [x,y,z,c]

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected input type of np.ndarray but got {type(image)}')
        return image.swapaxes(len(image.shape) - 2, 0)


class spekle:
    def __init__(self, gamma=.1):
        self.gamma = gamma

    def __call__(self, image):

        if not image.dtype == 'float':
            raise TypeError(f'Expected image datatype to be float but got {image.dtype}')
        if self.gamma > 1:
            raise ValueError(f'Maximum spekle gamma should be less than 1 [ gamma =/= {self.gamma} ]')

        image_shape = np.shape(image)
        noise = np.random.normal(0, self.gamma, image_shape)
        noise = np.float32(noise)
        image = image + noise
        image[image < 0] = 0
        image[image > 1] = 1

        return image


class random_gamma:
    def __init__(self, gamma_range=(.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, image: np.float):
        if not image.dtype == 'float':
            raise TypeError(f'Expected image dataype to be float but got {image.dtype}')

        factor = np.random.uniform(self.gamma_range[0], self.gamma_range[1], 1)
        if factor < 0:
            factor = 0
        return exposure.adjust_gamma(image, factor)


class random_affine:
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed):
        """
        Expects a list of numpy.ndarrays of the smame shape. Ranomly generates an affine
        tranformation matrix that transforms only the x,y dimmension
        and applies it to all images in list
        :param image_lsit:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list of images as input, but got {type(image)}')

        np.random.seed(seed)
        translation_x, translation_y = np.random.uniform(0, .5, size=2)

        # generate affine matrix
        mat = np.eye(image.ndim)
        mat[0, 1] = translation_x
        mat[1, 0] = translation_y

        return ndimage.affine_transform(image, mat, order=0, output_shape=image.shape, mode='reflect')


class random_rotate:
    def __init__(self, angle=None):
        self.angle = angle

    @joint_transform
    def __call__(self, image, seed):
        """
        Expects a list of numpy.ndarrays of all the same shape. Randonmly rotates the image along x or y dimmension
        and returns list of rotated images

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Expected list of images as input, but got {type(image)}')

        if not self.angle:
            np.random.seed(seed)
            theta = np.random.randint(0, 360, 1)[0]
        else:
            theta = self.angle

        return ndimage.rotate(image, angle=theta, reshape='false', order=0, mode='wrap', prefilter=False)


class normalize:
    def __init__(self, mean=[.5, .5, .5, .5], std=[.5, .5, .5, .5]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        shape = image.shape
        if image.ndim == 4:
            for channel in range(shape[-1]):
                image[:, :, :, channel] = (image[:, :, :, channel] - self.mean[channel]) / self.std[channel]
        if image.ndim == 3:
            for channel in range(shape[-1]):
                image[:, :, channel] = (image[:, :, channel] - self.mean[channel]) / self.std[channel]
        return image


class random_crop:
    def __init__(self, dim):
        self.dim = dim

    @joint_transform
    def __call__(self, image, seed):
        """
        Expect numpy ndarrays with color in the last channel of the array
        [x,y,z,c] or , [x,y,c]

        :param image:
        :return:
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Expected input to be list but got {type(image)}')

        np.random.seed(seed)
        if image.ndim == 4:  # 3D image
            shape = image.shape
            x = int(np.random.randint(0, shape[0] - self.dim[0] + 1, 1))
            y = int(np.random.randint(0, shape[1] - self.dim[1] + 1, 1))
            z = int(np.random.randint(0, shape[2] - self.dim[2] + 1, 1))
        elif image.ndim == 3:  # 3D image
            shape = image.shape
            x = np.random.randint(0, self.dim[0] - shape[0] + 1, 1)
            y = np.random.randint(0, self.dim[1] - shape[1] + 1, 1)
        else:
            raise ValueError(f'Expected np.ndarray with 3/4 ndims but found {image.ndim}')

        if not np.all(image.shape[0:-1:1] >= np.array(self.dim)):
            raise IndexError(f'Output dimmensions: {self.dim} are larger than input image: {shape}')

        if image.ndim == 4:  # 3D image
            out = image[x:x + self.dim[0] - 1:1, y:y + self.dim[1] - 1:1, z:z + self.dim[2] - 1:1, :]

        if image.ndim == 3:  # 3D image
            out = image[x:x + self.dim[0] - 1:1, y:y + self.dim[1] - 1:1, :]

        return out
