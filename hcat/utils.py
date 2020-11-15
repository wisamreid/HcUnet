import torch
import torchvision.ops

import numpy as np
from numba import njit
from numba import prange

import pandas as pd

import skimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage.transform
import skimage.feature

import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev

import pickle
import glob

import GPy

import matplotlib.pyplot as plt

from typing import Dict, Tuple, List


def pad_image_with_reflections(image: torch.Tensor, pad_size: Tuple[int] = (30, 30, 6)) -> torch.Tensor:
    """
    Pads image according to Unet spec
    expect [B, C, X, Y, Z]
    Adds pad size to each side of each dim. For example, if pad size is 10, then 10 px will be added on top, and on bottom.

    :param image:
    :param pad_size:
    :return:
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Expected image to be of type torch.tensor not {type(image)}')
    for pad in pad_size:
        if pad % 2 != 0:
            raise ValueError('Padding must be divisible by 2')

    image_size = image.shape
    pad_size = np.array(pad_size)

    left_pad = image.numpy()[:, :, pad_size[0]-1::-1, :, :]
    left_pad = torch.as_tensor(left_pad.copy())
    right_pad = image.numpy()[:, :, -1:-pad_size[0]-1:-1, :, :]
    right_pad = torch.as_tensor(right_pad.copy())
    image = torch.cat((left_pad, image, right_pad), dim=2)

    left_pad = 0
    right_pad = 0

    bottom_pad = image.numpy()[:, :, :, pad_size[1]-1::-1, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, :, :, -1:-pad_size[1]-1:-1, :]
    top_pad = torch.as_tensor(top_pad.copy())
    image = torch.cat((bottom_pad, image, top_pad), dim=3)
    bottom_pad = 0
    top_pad = 0

    bottom_pad = image.numpy()[ :, :, :, :, pad_size[2]-1::-1]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, :, :, :, -1:-pad_size[2]-1:-1]
    top_pad = torch.as_tensor(top_pad.copy())

    return torch.cat((bottom_pad, image, top_pad), dim=4)


def calculate_indexes(pad_size: int, eval_image_size: int, image_shape: int, padded_image_shape: int) -> List[List[int]]:
    """
    This calculates indexes for the complete evaluation of an arbitrarily large image by unet.
    each index is offset by eval_image_size, but has a width of eval_image_size + pad_size * 2.
    Unet needs padding on each side of the evaluation to ensure only full convolutions are used
    in generation of the final mask. If the algorithm cannot evenly create indexes for
    padded_image_shape, an additional index is added at the end of equal size.

    :param pad_size: int corresponding to the amount of padding on each side of the
                     padded image
    :param eval_image_size: int corresponding to the shape of the image to be used for
                            the final mask
    :param image_shape: int Shape of image before padding is applied
    :param padded_image_shape: int Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size > image_shape:
        return [[0, image_shape]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(f'Calculate_indexes has incorrect values {pad_size} | {image_shape} | {eval_image_size}:\n'
                           f'You are likely trying to have a chunk smaller than the set evaluation image size. '
                           'Please decrease number of chunks.')
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1])
        z2 = int(z-1) + (2 * pad_size)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size+pad_size * 2), padded_image_shape])
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + pad_size * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind


def get_cochlear_length(image, equal_spaced_distance, diagnostics=False):
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
    image = skimage.transform.downscale_local_mean(image, (10, 10)) > 0
    image = skimage.morphology.binary_closing(image)

    image = skimage.morphology.diameter_closing(image, 10)

    for i in range(5):
        image = skimage.morphology.binary_erosion(image)

    image = skimage.morphology.skeletonize(image)

    # first reshape to a logical image format and do a max project
    if image.ndim > 2:
        image = image.transpose((1,2,3,0)).mean(axis=3)/2**16
        image = skimage.exposure.adjust_gamma(image[:,:,2], .2)
        image = skimage.filters.gaussian(image, sigma=2) > .5
        image = skimage.morphology.binary_erosion(image)

    # Sometimes there are NaN or inf we have to take care of
    image[np.isnan(image)] = 0
    try:
        center_of_mass = np.array(scipy.ndimage.center_of_mass(image))
        while image[int(center_of_mass[0]), int(center_of_mass[1])] > 0:
            center_of_mass += 1
    except ValueError:
        center_of_mass = [image.shape[0], image.shape[1]]


    # Turn the binary image into a list of points for each pixel that isnt black

    x, y = image.nonzero()
    x += -int(center_of_mass[0])
    y += -int(center_of_mass[1])

    # Transform into spherical space
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)

    # sort by theta
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]

    # there will be a break somewhere because the cochlea isnt a full circle
    # Find the break and subtract 2pi to make the fun continuous
    loc = np.abs(theta[0:-2:1] - theta[1:-1:1])

    theta[loc.argmax()::] += -2*np.pi
    ind = theta.argsort()[1:-1:1]
    theta = theta[ind]
    r = r[ind]


    # seems to work better if we downsample by interpolation
    # theta_i =np.linspace(theta.min(), theta.max(), 100)
    # r_i= np.interp(theta_i, theta, r)
    # theta = theta_i
    # r = r_i

    # r_i = np.linspace(r.min(), r.max(), 200)
    # theta_i = np.interp(r_i, r, theta)
    # theta = theta_i
    # r = r_i

    # run a spline in spherical space after sorting to get a best approximated fit
    tck, u = splprep([theta, r], w=np.ones(len(r))/len(r), s=1.5e-6, k=3)
    u_new = np.arange(0,1,1e-4)

    # get new values of theta and r for the fitted line
    theta_, r_ = splev(u_new, tck)

    # plt.plot(theta, r, 'k.')
    # plt.xlabel('$\Theta$ (Radians)')
    # plt.ylabel('Radius')
    # plt.plot(theta_, r_)
    # plt.show()

    kernel = GPy.kern.RBF(input_dim=1, variance=100., lengthscale=5.)
    m = GPy.models.GPRegression(theta[:,np.newaxis], r[:,np.newaxis], kernel)
    m.optimize()
    r_, _ = m.predict(theta[:,np.newaxis])
    r_ = r_[:,0]
    theta_ = theta

    x_spline = r_*np.cos(theta_) + center_of_mass[1]
    y_spline = r_*np.sin(theta_) + center_of_mass[0]

    # x_spline and y_spline have tons and tons of data points.
    # We want equally spaced points corresponding to a certain distance along the cochlea
    # i.e. we want a point ever mm which is not guaranteed by x_spline and y_spline
    equal_spaced_points = []
    for i, coord in enumerate(zip(x_spline, y_spline)):
        if i == 0:
            base = coord
            equal_spaced_points.append(base)
        if np.sqrt((base[0] - coord[0])**2 + (base[1] - coord[1])**2) > equal_spaced_distance:
            equal_spaced_points.append(coord)
            base = coord

    equal_spaced_points = np.array(equal_spaced_points) * 10  # <-- Scale factor from above
    equal_spaced_points = equal_spaced_points.T

    curve = tck[1][0]
    if curve[0] > curve[-1]:
        apex = equal_spaced_points[:,-1]
        percentage = np.linspace(1,0,len(equal_spaced_points[0,:]))
    else:
        apex = equal_spaced_points[:,0]
        percentage = np.linspace(0,1,len(equal_spaced_points[0,:]))

    if not diagnostics:
        return equal_spaced_points, percentage, apex
    else:
        return equal_spaced_points, x_spline, y_spline, image, tck, u


def reconstruct_mask(path):
    """
    Assume we dont know anything about the number of pieces or the size of the image

    :param path:
    :return:
    """
    if path[-1] != '/':
        path = path + '/'
    files = glob.glob(path+'*.maskpart')
    x_max = 0
    y_max = 0

    if not files:
        raise FileExistsError('No Valid Part Files Found.')

    #Infer the size of the base mask
    for f in files:
        part = pickle.load(open(f, 'rb'))
        if part.loc[0] + part.shape[2] > x_max:
            x_max = part.loc[0] + part.shape[2]
        if part.loc[1] + part.shape[3] > y_max:
            y_max = part.loc[1] + part.shape[3]

    mask = np.ones((1, 1, x_max, y_max, part.shape[-1]), dtype=part.dtype)

    for f in files:
        part = pickle.load(open(f, 'rb'))
        x1 = part.loc[0]
        x2 = part.loc[0]+part.shape[2]
        y1 = part.loc[1]
        y2 = part.loc[1]+part.shape[3]
        # print(f'Index: [{x1}:{x2}, {y1}:{y2}]')
        mask[:, :, x1:x2, y1:y2, :] = part.mask.astype(part.dtype)

    return mask


def reconstruct_segmented(path):
    """
    Assume we dont know anything about the number of pieces or the size of the image

    :param path:
    :return:
    """
    if path[-1] != '/':
        path = path + '/'
    files = glob.glob(path+'*.maskpart')
    x_max = 0
    y_max = 0

    if not files:
        raise FileExistsError('No Valid Part Files Found.')

    #Infer the size of the base mask
    for f in files:
        part = pickle.load(open(f, 'rb'))
        if part.loc[0] + part.shape[2] > x_max:
            x_max = part.loc[0] + part.shape[2]
        if part.loc[1] + part.shape[3] > y_max:
            y_max = part.loc[1] + part.shape[3]

    mask = np.ones((1, 1, x_max, y_max, part.shape[-1]), dtype=part.dtype)
    max_id = 0
    for f in files:
        part = pickle.load(open(f, 'rb'))
        x1 = part.loc[0]
        x2 = part.loc[0]+part.shape[2]
        y1 = part.loc[1]
        y2 = part.loc[1]+part.shape[3]
        unique = part.segmented_mask.astype(part.dtype)
        unique[unique !=0] = (unique[unique != 0] + max_id)
        max_id = unique.max()
        mask[:, :, x1:x2, y1:y2, :] = unique
        del unique

    return mask


def merge_cell_candidates(candidate_list: dict, candidate_new: dict, initial_coords=(0, 0)):
    """
    Works when use nms... Im dumb

    Takes in a dict of current candidates and a dict of new candidates. From new candidates, adds
    :param candidate_list:
    :param candidate_new:
    :return:
    """
    iou_max = 0.20
    candidate_new['boxes'][:, [0, 2]] += initial_coords[1]
    candidate_new['boxes'][:, [1, 3]] += initial_coords[0]

    # Add the two dicts together
    candidate_list['boxes'] = torch.cat((candidate_list['boxes'], candidate_new['boxes']))
    candidate_list['scores'] = torch.cat((candidate_list['scores'], candidate_new['scores']))
    candidate_list['labels'] = torch.cat((candidate_list['labels'], candidate_new['labels']))
    candidate_list['z_level'] = torch.cat((candidate_list['z_level'], candidate_new['z_level']))

    keep = torchvision.ops.nms(boxes=candidate_list['boxes'],
                               scores=candidate_list['scores'],
                               iou_threshold=iou_max)

    candidate_list['boxes'] = candidate_list['boxes'][keep, :]
    candidate_list['scores'] = candidate_list['scores'][keep]
    candidate_list['labels'] = candidate_list['labels'][keep]
    candidate_list['z_level'] = candidate_list['z_level'][keep]

    #We gotta adjust boxes so that its with propper chunk offset

    return candidate_list


def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_box_pred(image, output,thr=.90):

    c = ['nul','r','b','y','w']

    boxes = output[0]['boxes'].detach().cpu().numpy().tolist()
    labels = output[0]['labels'].detach().cpu().int().numpy().tolist()
    scores = output[0]['scores'].detach().cpu().numpy().tolist()
    image = image.cpu()

    # x1, y1, x2, y2
    inp = image.numpy().transpose((1, 2, 0))
    mean = 0.5 #np.array([0.5, 0.5, 0.5])
    std = 0.5 #np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if inp.shape[-1] == 1:
        inp = inp[:,:,0]
        plt.imshow(inp, origin='lower', cmap='Greys_r')
    else:
        plt.imshow(inp, origin='lower', cmap='Greys_r')
    plt.tight_layout()

    for i, box in enumerate(boxes):

        if scores[i] < thr:
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1,x2],[y2,y2],c[labels[i]],lw=0.5)
        plt.plot([x1,x2],[y1,y1],c[labels[i]], lw=0.5)
        plt.plot([x1,x1],[y1,y2],c[labels[i]], lw=0.5)
        plt.plot([x2,x2],[y1,y2],c[labels[i]], lw=0.5)

    # plt.savefig('test.png',dp=1000)
    # plt.show()


def show_box_pred_simple(image, boxes):

    c = ['nul','r','b','y','w']

    # x1, y1, x2, y2


    plt.imshow(image,origin='lower')
    plt.tight_layout()

    for i, box in enumerate(boxes):

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1,x2],[y2,y2],'r', lw=0.5)
        plt.plot([x1,x2],[y1,y1],'r', lw=0.5)
        plt.plot([x1,x1],[y1,y2],'r', lw=0.5)
        plt.plot([x2,x2],[y1,y2],'r', lw=0.5)

    # plt.savefig('test.png',dp=1000)
    # plt.show()


def construct_instance_mask(cell_list: list, mask):
    unique_mask = torch.tensor(mask).type(torch.int)

    for i,cell in enumerate(cell_list):
        if cell.is_bad:
            continue

        index = cell.image_coords
        cell_mask = cell.unique_mask * (i + 1)
        cell_mask = torch.tensor(cell_mask).int()
        print(index[2]-index[0], index[3]-index[1], unique_mask.shape, index, mask.shape)
        unique_mask[0, 0, index[0]:index[2], index[1]:index[3], :][cell_mask > 0] = cell_mask[cell_mask > 0]

    return unique_mask


@njit(parallel=True)
def mask_to_lines(image:np.ndarray):
    """
    njit function for removing inner bits of segmentation mask
    Usefull if you only want an OUTLINE of the mask for overlay on the actuall image


    :param image:
    :return: zero_ind: boolean array of index's
    """
    y_lim = image.shape[-2]
    x_lim = image.shape[-3]
    x_ = x_lim-1
    y_ = y_lim-1
    ind = np.zeros(image.shape, dtype=np.bool_)

    # Loop over every pixel in image, if it isnt zero and matches its neightbors make ind TRUE
    # We'll remove True ind's later

    for z in prange(image.shape[-1]):
        for y in range(y_lim):
            if y == 0 or y == y_:
                continue
            for x in range(x_lim):
                if x == 0 or x == x_:
                    continue

                pix = image[0,0,x,y,z]
                if pix == 0:
                    continue

                left = image[0, 0, x-1, y, z]
                right = image[0, 0, x + 1, y, z]
                top = image[0, 0, x, y - 1, z]
                bottom = image[0, 0, x, y + 1, z]

                ind[0,0,x,y,z] = (left == pix and right == pix and top == pix and bottom == pix)

    return ind


def color_from_ind(i):
    """
    Take in some number and always generate a unique color from that number.
    Quick AF
    :param i:
    :return:
    """
    np.random.seed(i)
    return np.random.random(4)/.5


def cells_to_csv(all_cells: list, file_name: str) -> None:

    centers = []
    percent_location = []
    unique_id = []
    mean_gfp = []
    volume = []
    df = {}

    for cell in all_cells:
        centers.append(cell.center)
        percent_location.append(cell.distance_from_apex)
        unique_id.append(cell.unique_id)
        mean_gfp.append(cell.gfp_stats['mean'])
        volume.append(cell.volume)

    df = {'center':centers,
          'unique_id': unique_id,
          'percent_location': percent_location,
          'mean_gfp': mean_gfp,
          'volume': volume}

    df = pd.DataFrame(df)
    df.sort_values(by=['percent_location'])
    df.to_csv(file_name)
    return None























