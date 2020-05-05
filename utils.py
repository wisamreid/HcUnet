import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.filters
from scipy import interpolate
from skimage.morphology import skeletonize
import scipy.ndimage
import transforms as t
import torch.nn.functional as F

from scipy.interpolate import splprep, splev


def pad_image_with_reflections(image, pad_size=(30, 30, 6)):
    """
    Pads image according to Unet spec
    expect [B, C, X, Y, Z]

    :param image:
    :param pad_size:
    :return:
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Expected image to be of type torch.tensor not {type(image)}')
    for pad in pad_size:
        if pad % 2 != 0:
            raise ValueError('Padding must be divisible by 2')

    # image_size = image.shape  # should be some flavor of [Batchsize, C, X, Y, Z]
    #
    # out_size = [image_size[0],  # Batch Size
    #             image_size[1],  # Color Channels
    #             image_size[2] + pad_size[0],  # x
    #             image_size[3] + pad_size[1],  # y
    #             image_size[4] + pad_size[2],  # z
    #             ]
    #
    # left_pad = image[:, :, 0:pad_size[0], :, :].flip(2)
    # right_pad = image[:, :, -pad_size[0]::, :, :].flip(2)
    #
    # image = torch.cat((left_pad, image, right_pad), dim=2)
    #
    # bottom_pad = image[:, :, :, 0:pad_size[1], :].flip(3)
    # top_pad = image[:, :, :, -pad_size[1]::, :].flip(3)
    #
    # image = torch.cat((bottom_pad, image, top_pad), dim=3)
    #
    # bottom_pad = image[:, :, :, :, 0:pad_size[2]].flip(4)
    # top_pad = image[:, :, :, :, -pad_size[2]::].flip(4)

    image_size = image.shape  # expect x,y,z,c
    pad_size = np.array(pad_size)//2

    out_size = [
                image_size[0] + pad_size[0],  # x
                image_size[1] + pad_size[1],  # y
                image_size[2] + pad_size[2],  # z
                image_size[3]
                ]

    left_pad = image.numpy()[pad_size[0]-1::-1, :, :, :]
    left_pad = torch.as_tensor(left_pad.copy())
    right_pad = image.numpy()[-1:-pad_size[0]-1:-1, :, :, :]
    right_pad = torch.as_tensor(right_pad.copy())
    image = torch.cat((left_pad, image, right_pad), dim=0)

    bottom_pad = image.numpy()[:, pad_size[1]-1::-1, :, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, -1:-pad_size[1]-1:-1, :, :]
    top_pad = torch.as_tensor(top_pad.copy())
    image = torch.cat((bottom_pad, image, top_pad), dim=1)
    bottom_pad=0
    top_pad=0

    bottom_pad = image.numpy()[ :, :, pad_size[2]-1::-1, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, :, -1:-pad_size[2]-1:-1, :]
    top_pad = torch.as_tensor(top_pad.copy())

    return torch.cat((bottom_pad, image, top_pad), dim=2)



def predict_mask(model, image, device):
    """
    Takes in a model and an image and applies the model to all parts of the image.

    ALGORITHM:
    apply padding ->
    calculate indexes for unet based on image.shape ->
    apply Unet on slices of image based on indexes ->
    take only valid portion of the middle of each output mask ->
    construct full valid mask ->
    RETURN mask

    :param model: Trained Unet Model from unet.py
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :return mask:

    """
    PAD_SIZE = (100, 100, 6)
    print(f'SETTING SLICE TO {(image.shape[2]+PAD_SIZE[2] )}->{(image.shape[2]+PAD_SIZE[2] )// 2}')
    EVAL_IMAGE_SIZE = (500, 500, 46)#image.shape[2])

    mask = torch.zeros((1,1,image.shape[0],image.shape[1], image.shape[2]), dtype=np.bool)
    im_shape = image.shape
    # Apply Padding
    image = pad_image_with_reflections(torch.as_tensor(image), pad_size=PAD_SIZE)
    #  We now calculate the indicies for our image
    x_ind = calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], image.shape[0])
    y_ind = calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], image.shape[1])
    z_ind = calculate_indexes(PAD_SIZE[2], EVAL_IMAGE_SIZE[2], im_shape[2], image.shape[2])

    iterations = 0
    max = len(x_ind) * len(y_ind) * len(z_ind)
    # Loop and apply unet
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            for k, z in enumerate(z_ind):
                print(f'\r{iterations}/{max} ',end=' ')
                padded_image_slice = t.to_tensor()(image[x[0]:x[1], y[0]:y[1]:, z[0]:z[1], :].numpy())
                print(padded_image_slice.shape, end='')
                # padded_image_slice = image[:,:,x[0]:x[1], y[0]:y[1]:, z[0]:z[1]]
                with torch.no_grad():
                    valid_out = model(padded_image_slice.float().to(device))

                valid_out = valid_out[:,:,
                                     PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                                     PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                                     PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2],
                                         ]

                #do everthing in place to save memory
                F.relu_(valid_out)
                valid_out.gt_(.5)

                mask[:, :, x[0]:x[0]+valid_out.shape[2],
                           y[0]:y[0]+valid_out.shape[3],
                           z[0]:z[0]+valid_out.shape[4]] = valid_out

                iterations += 1

    return mask


def calculate_indexes(pad_size, eval_image_size, image_shape, dim_shape):

    ind_list = torch.arange(pad_size, image_shape, eval_image_size)
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1])-pad_size
        z2 = int(z-1)+pad_size
        ind.append([z1, z2])
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([dim_shape - (eval_image_size+pad_size*2), dim_shape])
    else:
        z1 = dim_shape-(eval_image_size + pad_size*2)
        z2 = dim_shape-1
        ind.append([z1, z2])
    return ind


def get_cochlear_length(image, calibration, diagnostics=False):

    #JUST DO IT ON THE MASK YOU BOOB. WONT HAVE TO WORRY ABOUT NONSENSE THRESHOLDING

    """
    Input an image ->
    max project ->
    reduce image size ->
    run b-spline fit of resulting data on myosin channel ->
    return array of shape [2, X] where [0,:] is x and [1,:] is y
    and X is ever mm of image

    IMAGE: numpy image
    CALIBRATION: calibration info

    :return: Array
    """
    og_image = np.copy(image)
    # first reshape to a logical image format and do a max project
    image = image.transpose((1,2,3,0)).mean(axis=3)/2**16

    plt.imshow(image[:,:,1:4])
    plt.show()

    image = skimage.exposure.adjust_gamma(image[:,:,2], .2)
    image = skimage.filters.gaussian(image, sigma=2) > .5
    image = skimage.morphology.binary_erosion(image)
    com = scipy.ndimage.center_of_mass(image)

    plt.imshow(image)
    plt.show()

    x, y = image.nonzero()
    x += -int(com[0])
    y += -int(com[1])
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]
    loc = np.abs(theta[0:-2:1] - theta[1:-1:1])

    plt.plot(theta, r, 'k.')
    plt.axvline(theta[loc.argmax()])
    plt.show()

    theta[loc.argmax()::] += -2*np.pi
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]

    plt.plot(theta, r,'k.')
    plt.axvline(theta[loc.argmax()])
    plt.show()
    # m = len(theta)
    tck, u = splprep([theta, r], w=np.ones(len(r))/len(r) , s=0.01, k=3)
    u_new = np.arange(0,1,1e-4)
    theta_, r_ = splev(u_new, tck)

    plt.plot(theta, r, 'k.')
    plt.plot(theta_, r_, 'r')
    plt.show()

    x_spline = r_*np.cos(theta_) + com[1]
    y_spline = r_*np.sin(theta_) + com[0]
    equal_spaced_points = []
    for i, coord in enumerate(zip(x_spline, y_spline)):
        if i == 0:
            base = coord
            equal_spaced_points.append(base)
        if np.sqrt((base[0] - coord[0])**2 + (base[1] - coord[1])**2) > calibration:
            equal_spaced_points.append(coord)
            base = coord

    equal_spaced_points = np.array(equal_spaced_points)
    print(equal_spaced_points.shape)

    plt.plot(x+com[0],y+com[1],'k.')
    plt.plot(y_spline, x_spline,'r')
    plt.show()

    # plt.plot(y,x,'k.')
    # plt.plot(r*np.cos(theta), r*np.sin(theta), 'r')
    plt.plot(y+com[1],x+com[0],'k.')
    plt.plot(x_spline, y_spline, 'r')
    plt.plot(equal_spaced_points[:,0], equal_spaced_points[:,1], 'bx')
    plt.show()

    # plt.plot(r*np.cos(theta), r*np.sin(theta), 'r')
    plt.imshow(skimage.exposure.adjust_gamma(og_image.mean(axis=0)[:,:,0:3]/2**16,gamma= .2))
    plt.plot(x_spline, y_spline, 'r')
    plt.plot(equal_spaced_points[:,0], equal_spaced_points[:,1], 'bx')
    plt.show()

    if not diagnostics:
        return equal_spaced_points.T
    else:
        return equal_spaced_points.T, x_spline, y_spline, image, tck, u



