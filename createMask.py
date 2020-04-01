import skimage.io as io
import numpy as np


ogimage = io.imread(
    '/Users/chrisbuswinka/AI-Lab Dropbox/Unet/Data/C3-Jan-27-AAV2-PHPB-m5.lif---TileScan-m5-1.labels.tif'
                 )

testind = [9,439,109]
# image = np.copy(ogimage[:,85:115,423:435, :]/2**16)
image = np.copy(ogimage)
background = np.copy(image[0, 0, 0, :])

# plt.imshow(image[9,:,:,:])
# plt.title('Original')
# plt.show()

for z in range(image.shape[0]):
    for x in range(image.shape[1]):
        for y in range(image.shape[2]):

            if np.all(image[z, x, y, :] == background):
                image[z, x, y, :] = [0, 0, 0]

background = np.copy(image[0, 0, 0, :])

# plt.imshow(image[9,:,:,:])
# plt.title('Background Removed')
# plt.show()
#
for nul in range(1):
    for z in range(image.shape[0] - 1):
        if z == 0:
            continue

        for x in range(image.shape[1] - 1):
            if x == 0:
                continue

            for y in range(image.shape[2] - 1):
                if y == 0:
                    continue
                pixel = np.copy(image[z, x, y, :])

                if np.all(pixel == background):
                    continue

                # print(f'C: {z,y,x}| P:    {pixel} ')
                # print(f'\t[z, x + 1, y, :] {image[z, x + 1, y, :]} -> {np.any(image[z, x + 1, y, :] != pixel)}')
                # print(f'\t[z, x - 1, y, :] {image[z, x - 1, y, :]} -> {np.any(image[z, x - 1, y, :] != pixel)}')
                # print(f'\t[z, x, y + 1, :] {image[z, x, y + 1, :]} -> {np.any(image[z, x, y + 1, :] != pixel)}')
                # print(f'\t[z, x, y - 1, :] {image[z, x, y - 1, :]} -> {np.any(image[z, x, y - 1, :] != pixel)}')
                # print(f'\t[z + 1, x, y, :] {image[z + 1, x, y, :]} -> {np.any(image[z + 1, x, y, :] != pixel)}')
                # print(f'\t[z - 1, x, y, :] {image[z - 1, x, y, :]} -> {np.any(image[z - 1, x, y, :] != pixel)}')

                if np.any(image[z, x + 1, y, :] != pixel) and not np.all(image[z, x + 1, y, :] == background):
                    # print(f'C: {z,y,x}| P: {pixel} -- {image[z, x + 1, y, :]}')
                    image[z, x, y, :] = background
                if np.any(image[z, x - 1, y, :] != pixel) and not np.all(image[z, x - 1, y, :] == background):
                    image[z, x, y, :] = background
                    # print(f'C: {z,y,x}| P: {pixel} -- {image[z, x - 1, y, :]}')

                if np.any(image[z, x, y + 1, :] != pixel) and not np.all(image[z, x, y + 1, :] == background):
                    # print(f'C: {z,y,x}| P: {pixel} -- {image[z, x, y+1, :]}')
                    image[z, x, y, :] = background
                if np.any(image[z, x, y - 1, :] != pixel) and not np.all(image[z, x, y - 1, :] == background):
                    # print(f'C: {z,y,x}| P: {pixel} -- {image[z, x, y-1, :]}')
                    image[z, x, y, :] = background

                # if np.any(image[z + 1, x, y, :] != pixel) and not np.all(image[z + 1, x, y, :] == background):
                #     # print(f'C: {z,y,x}| P: {pixel} -- {image[z+1, x, y, :]}')
                #     image[z, x, y, :] = background
                # if np.any(image[z - 1, x, y, :] != pixel) and not np.all(image[z - 1, x, y, :] == background):
                #     # print(f'C: {z,y,x}| P: {pixel} -- {image[z-1, x, y, :]}')
                #     image[z, x, y, :] = background


    # plt.imshow(image[9,:,:,:])
    # plt.title('Background Removed'+str(nul))
    # plt.show()

# for z in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for y in range(image.shape[2]):
#
#             if np.any(image[z, x, y, :] != background):
#                 image[z, x, y, :] = [1, 1, 1]


io.imsave('colo_test.tif', image.astype(np.uint16))


