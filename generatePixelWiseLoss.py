import numpy as np
import skimage.io as io
from numba import njit
import matplotlib.pyplot as plt

mask = io.imread('Data/colo_test.tif')  # np.uint16
ms = mask.shape
bigmask = np.zeros((ms[0], ms[1] + 100, ms[2] + 100, ms[3]))
bigmask[:, 50:ms[1] + 50, 50:ms[2] + 50, :] = mask
background = bigmask[0, 0, 0, :]  # np.uint16

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

radius = zip([1, 0, -1, 0, 1, -1, -1, 1],
             [0, 1, 0, -1, 1, 1, -1, -1])


@njit
def find_closest(bigmask, z, y, x):
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
                    # print(lens, w0 * np.exp(-1 * ((lens[0] + lens[1]) ** 2) / (2 * (sigma ** 2))))
                    return w0 * np.exp(-1 * ((lens[0] + lens[1]) ** 2) / (2 * (sigma ** 2)))

    return 1


# @njit
def loop(pwl, bigmask, image_shape, find_closest):
    for z in range(image_shape[0]):
        for y in range(image_shape[1]):
            for x in range(image_shape[2]):
                if np.all(bigmask[z, y, x, :] == np.array([0, 0, 0])):
                    pwl[z, y, x] = find_closest(bigmask, int(z), int(y), int(x))

    return pwl


pwl = loop(pwl, bigmask, image_shape, find_closest)
pwl = pwl[:, 50:-50:1, 50:-50:1]


io.imsave('pwl_a.tif', pwl.astype(np.float64))











