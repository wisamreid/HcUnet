import numpy as np
import scipy.ndimage.morphology
import scipy.ndimage
import skimage.io as io
import skimage.morphology
from numba import njit


class makePWL:
    def __call__(self, imagepath):
        """
        Pixel wize loss


        :param imagepath:
        :return:
        """
        print(f'STARTING PIXEL WISE LOSS')
        mask = io.imread(imagepath)  # np.uint16
        print(mask.dtype)
        ms = mask.shape
        bigmask = np.zeros((ms[0], ms[1] + 100, ms[2] + 100, ms[3]))
        bigmask[:, 50:ms[1] + 50, 50:ms[2] + 50, :] = mask
        background = bigmask[0, 0, 0, :]  # np.uint16
        print(background)

        # for z in range(mono_bigmask.shape[0]):
        ##    for x in range(mono_bigmask.shape[1]):
        #        for y in range(mono_bigmask.shape[2]):
        #
        #            if np.any(mono_bigmask[z, x, y, :] != background):
        #                mono_bigmask[z, x, y, :] = np.array([2**16, 2**16, 2**16], dtype=np.uint16)

        # Get one slice and invert 1 and 0
        # BEFORE: Cells are 1
        # pwl = np.copy(mono_bigmask)[:,:,:,0]  # np.uint8 [0 or 1]
        # pwl[pwl == 0] = 1
        # pwl[pwl == 2**16-1] = 0
        # Cells are 0
        pwl = np.zeros((bigmask.shape[0:-1:1]), dtype=np.float64)
        image_shape = bigmask.shape
        pwl = self.loop(pwl, bigmask, image_shape)
        print(f'PWL MAX {pwl.max()}')
        pwl = pwl[:, 50:-50:1, 50:-50:1]
        print(f'PWL MAX 2 {pwl.max()}')
        pwl_add = pwl + 1

        return pwl

    def loop(self, pwl, bigmask, image_shape):
        for z in range(image_shape[0]):
            print(z)
            for y in range(image_shape[1]):
                for x in range(image_shape[2]):
                    if np.all(bigmask[z, y, x, :] == np.array([0, 0, 0])):
                        pwl[z, y, x] = self.find_closest(bigmask, pwl, int(z), int(y), int(x))

        return pwl

    @staticmethod
    @njit
    def find_closest(bigmask, pwl,  z, y, x):
        angles = np.linspace(0, 2 * np.pi, 63)
        closest = np.array([0, 0, 0], dtype=np.float64)
        nclosest = np.array([0, 0, 0], dtype=np.float64)
        lens = []
        w0 = 11
        sigma = 5

        for i, l in enumerate(np.arange(1, 10)):
            for theta in angles:

                dx = int(np.rint(l * np.cos(theta)))
                dy = int(np.rint(l * np.sin(theta)))

                if y + dy < pwl.shape[1] and x + dx < pwl.shape[2]:
                    if np.any(bigmask[z, y + dy, x + dx] != np.array([0, 0, 0])):

                        if np.all(closest == np.array([0, 0, 0])):

                            closest = bigmask[z, y + dy, x + dx, :]
                            lens.append(l)

                        elif np.all(nclosest == np.array([0, 0, 0])) and np.any(closest != np.array([0, 0, 0])):

                            if np.any(bigmask[z, y + dy, x + dx, :] != closest):
                                nclosest = bigmask[z, y + dy, x + dx, :]
                                lens.append(l)

                    if np.any(closest != np.array([0, 0, 0])) and np.any(nclosest != np.array([0, 0, 0])):
                        return w0 * np.exp(-1 * ((lens[0] + lens[1]) ** 2) / (2 * (sigma ** 2)))

        return 0


class makeMask:
    def __init__(self, erosion=False):
        self.erosion = erosion

    def __call__(self, imagepath):
        # EVERYTHING IS [Z,Y,X,C]
        ogimage = io.imread(imagepath)

        image = np.copy(ogimage)
        background = np.copy(image[0, 0, 0, :])

        image = self.set_background(image, background)

        background = np.copy(image[0, 0, 0, :])

        image = self.create_mask(image, background)

        if self.erosion:
            binary_mask = colormask_to_mask(image)
            erroded_mask = np.zeros(binary_mask.shape)
            for i in range(binary_mask.shape[0]):
                erroded_mask[i,:,:] = scipy.ndimage.morphology.binary_erosion(binary_mask[i,:,:])
                erroded_mask = erroded_mask >= 1
            ogimage[erroded_mask == 0] = background
            image = np.copy(ogimage)

        return image

    @staticmethod
    @njit
    def set_background( image, background):
        for z in range(image.shape[0]):
            for x in range(image.shape[1]):
                for y in range(image.shape[2]):

                    if np.all(image[z, x, y, :] == background):
                        image[z, x, y, :] = [0, 0, 0]
        return image

    @staticmethod
    @njit
    def create_mask(image, background):

        for nul in range(1):
            for z in range(image.shape[0] - 1):

                for x in range(image.shape[1] - 1):
                    if x == 0:
                        continue

                    for y in range(image.shape[2] - 1):
                        if y == 0:
                            continue
                        pixel = np.copy(image[z, x, y, :])

                        if np.all(pixel == background):
                            continue

                        if np.any(image[z, x + 1, y, :] != pixel) and not np.all(image[z, x + 1, y, :] == background):
                            image[z, x, y, :] = background
                        if np.any(image[z, x - 1, y, :] != pixel) and not np.all(image[z, x - 1, y, :] == background):
                            image[z, x, y, :] = background
                        if np.any(image[z, x, y + 1, :] != pixel) and not np.all(image[z, x, y + 1, :] == background):
                            image[z, x, y, :] = background
                        if np.any(image[z, x, y - 1, :] != pixel) and not np.all(image[z, x, y - 1, :] == background):
                            image[z, x, y, :] = background
        return image


@njit
def colormask_to_mask(colormask):

    mask_shape = colormask.shape
    background = np.array([0,0,0])

    for z in range(mask_shape[0]):
        for y in range(mask_shape[1]):
            for x in range(mask_shape[2]):
                if np.any(colormask[z,y,x,:] != background):
                    colormask[z,y,x,:] = [1, 1, 1]

    return (colormask[:,:,:,0]*255).astype(np.uint8)


class CalculateCenterOfMass:
    def __init__(self):
        pass
    def __call__(self, imagepath):
        # EVERYTHING IS [Z,Y,X,C]
        ogimage = io.imread(imagepath)
        image = np.copy(ogimage)
        background = np.copy(image[0, 0, 0, :])

        image = self.set_background(image, background)

        unique_colors = np.unique(image.reshape(-1, image.shape[3]), axis=0)

        new_image = np.zeros(image.shape[:-1:], dtype=np.uint32)
        center_of_mass = np.zeros(new_image.shape)


        for id, color in enumerate(unique_colors):
            ix = np.ones(new_image.shape, dtype=np.bool)
            for c in range(image.shape[-1]):
                ix = np.logical_and(ix, image[:, :, :, c] == color[c])
            new_image[ix] = id

            center = scipy.ndimage.center_of_mass(new_image == id)
            x = int(np.round(center[2]))
            y = int(np.round(center[1]))
            z = int(np.round(center[0]))
            center_of_mass[z, y, x] = id

        return center_of_mass.astype(np.uint8), new_image

    @staticmethod
    @njit
    def set_background( image, background):
        for z in range(image.shape[0]):
            for x in range(image.shape[1]):
                for y in range(image.shape[2]):

                    if np.all(image[z, x, y, :] == background):
                        image[z, x, y, :] = [0, 0, 0]
        return image


class VectorToCenter():
    def __init__(self):
        pass

    def __call__(self, center, colormask, mask):
        # Stuff is in [z,y,x,c]
        print(len(np.unique(colormask)), len(np.unique(center)), colormask.max()+1)
        unique_cells = np.unique(colormask)
        vector = np.zeros([colormask.shape[0], colormask.shape[1], colormask.shape[2], 3])

        for id in unique_cells:
            if id == 0:
                continue

            idx = colormask == id
            com = np.where(center == id)

            indicies = np.where(idx)

            z = indicies[0]
            z_vec = indicies[0]

            y = indicies[1]
            y_vec = indicies[1]

            x = indicies[2]
            x_vec = indicies[2]

            z_vec = -z_vec + com[0]
            y_vec = -y_vec + com[1]
            x_vec = -x_vec + com[2]

            for i in range(len(z)):
                vector[z[i], y[i], x[i], :] = [z_vec[i], y_vec[i], x_vec[i]]

        return vector








