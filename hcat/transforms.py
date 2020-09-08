import torch
import scipy.ndimage as ndimage
import skimage.exposure as exposure
import skimage.transform as transform
import numpy as np
import copy
import glob
import skimage.io as io
import cv2

# DECORATOR
def joint_transform(func):
    """
    This wrapper generates a seed value for each transform and passes an image from a list to that function
    It then passes images from a list to the function one at a time and returns a list of outputs

    The wrapper is designed to work on the __call__ method of a class. It expects 'self' to be the first argument
    followed by image, then seed. It allows you to pass a list of 3D images to a function written to transform a
    single image.

    [image_1, image_2, image_3] -> joint_transform(fun) = [fun(image_i, seed), fun(image_2, seed), fun(image_3, seed)]

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


class to_float:
    """
    Takes a numpy image matrix of type uint8 or uint16 (standard leica confocal image conventions) and
    rescale to a float between 0 and 1.
    """
    def __init__(self):
        pass

    @joint_transform
    def __call__(self, image, seed=None):
        if image.dtype == 'uint16':
            image = image.astype(dtype='float', casting='same_kind', copy=False)
            image /= 2**16
        elif image.dtype == 'uint8':
            image = image.astype('float', copy=False, casting='same_kind')
            image /= 2**8
        elif image.dtype == 'float':
            pass
        else:
            raise TypeError('Expected image datatype of uint8 or uint16 ')
        return image


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
        image = torch.as_tensor(image, dtype=torch.half)
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
        """
        :param gamma: float | Maximum intensity of noise added to each channel
        """
        self.gamma = gamma

    def __call__(self, image):
        """
        :param image: np.ndarray(dtype=np.float)
        :return:
        """
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
        raise NotImplemented('hcat.transforms.random_affine is currently in an unusable state. Please do not use.')
        pass

    @joint_transform
    def __call__(self, image, seed):
        """
        Expects a list of numpy.ndarrays of the same shape. Randomly generates an affine
        transformation matrix that transforms only the x,y dimension
        and applies it to all images in list
        :param image:
        :param seed:
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

        out = ndimage.affine_transform(image.astype(np.float), mat, order=0, output_shape=image.shape, mode='reflect')
        return out.round()


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

        return ndimage.rotate(image.astype(np.float), angle=theta, reshape='false', order=0, mode='wrap', prefilter=False)


class normalize:
    def __init__(self, mean=[.5, .5, .5, .5], std=[.5, .5, .5, .5]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, list):
            image = image[0]
        shape = image.shape
        if image.ndim == 4:
            for channel in range(shape[-1]):
                image[:, :, :, channel] += -self.mean[channel]
                image[:, :, :, channel] /=  self.std[channel]
        elif image.ndim == 3:
            for channel in range(shape[-1]):
                image[:, :, channel] += -self.mean[channel]
                image[:, :, channel] /=  self.std[channel]
        else:
            raise ValueError(f'Expected a 3 or 4 dimensional image not: {image.ndim} with shape: {image.shape}')
        return image


class drop_channel:
    def __init__(self, chance):
        self.chance = chance

    def __call__(self, image):
        """
        assume in [x,y,z,c]
        :param image:
        :return:
        """
        if np.random.random() > self.chance:
            i = np.random.randint(0, image.shape[-1])
            image[:, :, :, i] = 0
        return image


class random_intensity:
        def __init__(self, range=(.5,1.2), chance=.75):
            self.range = range
            self.chance = chance

        def __call__(self, image):
            """
            assume in [x,y,z,c]
            :param image:
            :return:
            """
            # if np.random.random() > self.chance:
            #     range_min = np.random.randint(0, self.range[0]*10)/100
            #     range_max = np.random.randint(80, self.range[1]*100)/100
            #     image = exposure.rescale_intensity(image, 'image', (range_min, range_max))
            #     image[image > 1] = 1
            #     image[image < 0] = 0
            if image.ndim == 4:
                for c in range(image.shape[-1]):
                    if np.random.random() > self.chance:
                        val = np.random.randint(0, 50, 1)/100
                        image[:, :, :, c] -= val
                        image[image < 0] = 0
                        image[np.isnan(image)] = 0
                        image[np.isinf(image)] = 1
            elif image.ndim == 3:
                for c in range(image.shape[-1]):
                    if np.random.random() > self.chance:
                        val = np.random.randint(0, 50, 1)/100
                        image[:, :, c] -= val
                        image[image < 0] = 0
                        image[np.isnan(image)] = 0
                        image[np.isinf(image)] = 1

            return image


class random_crop:
    def __init__(self, dim):
        self.dim = np.array(dim)

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
        dim = copy.copy(self.dim)

        np.random.seed(seed)
        if image.ndim == 4:  # 3D image
            shape = image.shape[0:-1]
            ind = shape < dim
            for i, val in enumerate(ind):
                if val:
                    dim[i] = shape[i]

            x = int(np.random.randint(0, shape[0] - dim[0] + 1, 1))
            y = int(np.random.randint(0, shape[1] - dim[1] + 1, 1))
            if dim[2] > shape[2]:
                z = int(np.random.randint(0, shape[2] - dim[2] + 1, 1))
            else:
                z = 0
                dim[2] = shape[2]

        elif image.ndim == 3:  # 2D image
            shape = image.shape
            x = np.random.randint(0, dim[0] - shape[0] + 1, 1)
            y = np.random.randint(0, dim[1] - shape[1] + 1, 1)
        else:
            raise ValueError(f'Expected np.ndarray with 3/4 ndims but found {image.ndim}')

        if not np.all(image.shape[0:-1:1] >= np.array(dim)):
            raise IndexError(f'Output dimmensions: {dim} are larger than input image: {shape}')

        if image.ndim == 4:  # 3D image
            out = image[x:x + dim[0] - 1:1, y:y + dim[1] - 1:1, z:z + dim[2] - 1:1, :]

        if image.ndim == 3:  # 3D image
            out = image[x:x + dim[0] - 1:1, y:y + dim[1] - 1:1, :]

        return out


# Cant be a @joint_transform because it needs info from one image to affect transforms of other
class nul_crop:

    def __init__(self, rate=1):
        self.rate = rate

    # Cant be a @joint_transform because it needs info from one image to affect transforms of other
    def __call__(self, image_list: list) -> list:
        """
        IMAGE MASK PWL
        :param image:
        :return:
        """
        if not isinstance(image_list, list):
            raise ValueError(f'Expected input to be list but got {type(image_list)}')

        if np.random.random() < self.rate:
            out = []
            mask = image_list[1]
            lr = mask.sum(axis=1).sum(axis=1).flatten() > 1
            for i, im in enumerate(image_list):
                image_list[i] = im[lr, :, :, :]

            mask = image_list[1]
            ud = mask.sum(axis=0).sum(axis=1).flatten() > 1
            for i, im in enumerate(image_list):
                out.append(im[:, ud, :, :])
        else:
            out = image_list

        return out


#  FROM FASTER_RCNN CODE


class random_x_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, image, boxes):
        """
        Code specifically for transforming images and boxes for fasterRCNN
        Not Compatable with UNET

        :param image:
        :param boxes:
        :return:
        """
        flip = np.random.uniform(0,1,1) < self.rate

        shape = image.shape
        boxes = np.array(boxes)

        if flip:
            image = np.copy(image[::-1,:,:])
            boxes[:,1] = (boxes[:,1] * -1) + shape[0]
            boxes[:,3] = (boxes[:,3] * -1) + shape[0]

        newboxes = np.copy(boxes)
        # newboxes[:,0] = np.min(boxes[:,0:3:2],axis=1)
        newboxes[:,1] = np.min(boxes[:,1:4:2],axis=1)
        # newboxes[:,2] = np.max(boxes[:,0:3:2],axis=1)
        newboxes[:,3] = np.max(boxes[:,1:4:2],axis=1)
        boxes = np.array(newboxes,dtype=np.int64)

        return image, boxes.tolist()


class random_y_flip:
    def __init__(self, rate=.5):
        self.rate = rate

    def __call__(self, image, boxes):
        """
        FASTER RCNN ONLY

        :param image:
        :param boxes:
        :return:
        """

        flip = np.random.uniform(0, 1, 1) > 0.5

        shape = image.shape
        boxes = np.array(boxes,dtype=np.int)

        if flip:
            image = np.copy(image[:,::-1,:])
            boxes[:, 0] = (boxes[:, 0] * -1) + shape[1]
            boxes[:, 2] = (boxes[:, 2] * -1) + shape[1]

        newboxes = np.copy(boxes)
        newboxes[:,0] = np.min(boxes[:,0:3:2],axis=1)
        # newboxes[:,1] = np.min(boxes[:,1:4:2],axis=1)
        newboxes[:,2] = np.max(boxes[:,0:3:2],axis=1)
        # newboxes[:,3] = np.max(boxes[:,1:4:2],axis=1)
        boxes = np.array(newboxes,dtype=np.int64)

        return image, boxes.tolist()


class random_resize:
    def __init__(self, rate=.5, scale=(.8, 1.2)):
        self.rate = rate
        self.scale = scale

    def __call__(self, image, boxes):
        """
        FASTER RCNN ONLY


        :param image:
        :param boxes:
        :return:
        """

        scale = np.random.uniform(self.scale[0]*100, self.scale[1]*100, 1) / 100
        shape = image.shape

        new_shape = np.round(shape * scale)
        new_shape[2] = shape[2]

        image = transform.resize(image, new_shape)

        boxes = np.array(boxes) * scale
        boxes = np.round(boxes).round()
        boxes = np.array(boxes, dtype=np.int64)

        return image, boxes.tolist()


class remove_channel:

    def __init__(self, remaining_channel_index=[0,2,3]):
        self.index_remain = remaining_channel_index

    def __call__(self, image):
        """
        Assumes channels are at last dimmension of tensor!

        :param image:
        :return:
        """

        if image.shape[-1] == len(self.index_remain):
            return image

        if image.ndim == 3:
            image  = image[:,:,self.index_remain]
        elif image.ndim == 4:
            image  = image[:,:,:,self.index_remain]
        elif image.ndim == 5:
            image  = image[:,:,:,:,self.index_remain]

        return image


class clean_image:
    """
    Simple transform ensuring no nan values are passed to the model.

    """

    def __init__(self):
        pass
    
    @joint_transform
    def __call__(self, image, seed):
        type = image.dtype
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 1

        return image.astype(dtype=type)


class add_junk_image:
    """
    Simple transform ensuring no nan values are passed to the model.
    """

    def __init__(self, path, channel_index=[0, 2, 3], junk_image_size=(100, 100), normalize=False):
        """
        Take in a path to images that are p junk. Make sure the images are
        :param path:
        :param channel_index: if the image has more channels than channel index, than it reduces down
        """

        self.index_remain = channel_index
        self.path = path

        if self.path[-1] != '/':
            self.path += '/'

        self.files = glob.glob(self.path+'*.tif')

        if len(self.files) < 1:
            raise FileNotFoundError(f'No valid *.tif files found at {path}')

        self.junk_image_size = junk_image_size
        self.normalize = normalize

        if normalize:
            self.mean = normalize['mean']
            self.std = normalize['std']

        self.images = []
        for file in self.files:
            im = io.imread(file)
            im = to_float()(im)
            if self.normalize:
                for i in range(im.shape[-1]):
                    im[:, :, i] -= self.mean[i]
                    im[:, :, i] /= self.std[i]

            # Check to see if the image has the right number of color channels
            # If not, we drop one channel by the indexes defined by the user with self.index_remain
            if not im.shape[-1] == len(self.index_remain):
                im = im[:, :, self.index_remain]

            self.images.append(im)

    def __call__(self, image, boxes):
        """
        Loads a junk image and crops it randomly. Then adds it to image and removes boxes that overlap with
        the added region.

        :param image:
        :param boxes:
        :return:
        """
        file_index = np.random.randint(0, len(self.files))
        junk_image = self.images[file_index]

        shape = junk_image.shape

        try:
            x = np.random.randint(0, shape[0]-(self.junk_image_size[0]+1))
            y = np.random.randint(0, shape[1]-(self.junk_image_size[1]+1))
        except ValueError:
            raise ValueError(f'Junk image file {self.files[file_index]} has a shape ({shape}) smaller than max user defined shape')

        junk_image = junk_image[x:x+self.junk_image_size[0], y:y+self.junk_image_size[1], :]

        shape = image.shape
        x = np.random.randint(0, shape[0] - (self.junk_image_size[0] + 1))
        y = np.random.randint(0, shape[1] - (self.junk_image_size[1] + 1))
        image[x:x + self.junk_image_size[0], y:y + self.junk_image_size[1], :] = junk_image

        # VECTORIZE FOR SPEED????

        for i, box in enumerate(boxes):
            box = np.array(box)
            box_x = box[[0, 2]]
            box_y = box[[1, 3]]

            a = box_x < (x + self.junk_image_size[0])
            b = box_x > x
            c = np.logical_and(a, b)
            if np.any(c):
                boxes.pop(i)
                continue

            a = box_y < (y + self.junk_image_size[1])
            b = box_y > y
            c = np.logical_and(a, b)
            if np.any(c):
                boxes.pop(i)
                continue

        return image, boxes


def distance_transform(image):

    #Assume image is in a standard state from io.imread [Z, Y/X, X/Y, C]

    distance = np.zeros(image.shape, dtype=np.float)

    for i in range(image.shape[0]):
        distance[i, :, :, :] = cv2.distanceTransform(image[i, :, :, :], cv2.DIST_L2, 5)

    return distance