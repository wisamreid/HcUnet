import torch
import torchvision.ops
import numpy as np
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage
import skimage.feature
import scipy.ndimage
import scipy.ndimage.morphology
from scipy.interpolate import splprep, splev
import pickle
import glob
import matplotlib.pyplot as plt


def pad_image_with_reflections(image, pad_size=(30, 30, 6)):
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


def calculate_indexes(pad_size, eval_image_size, image_shape, padded_image_shape):
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
    :param image_shape: Shape of image before padding is applied
    :param padded_image_shape: Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

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
    # make a copy of the image
    og_image = np.copy(image)

    # first reshape to a logical image format and do a max project
    image = image.transpose((1,2,3,0)).mean(axis=3)/2**16

    image = skimage.exposure.adjust_gamma(image[:,:,2], .2)
    image = skimage.filters.gaussian(image, sigma=2) > .5
    image = skimage.morphology.binary_erosion(image)
    center_of_mass = scipy.ndimage.center_of_mass(image)

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
    ind = theta.argsort()
    theta = theta[ind]
    r = r[ind]

    # run a spline in spherical space after sorting to get a best approximated fit
    tck, u = splprep([theta, r], w=np.ones(len(r))/len(r), s=0.01, k=3)
    u_new = np.arange(0,1,1e-4)

    # get new values of theta and r for the fitted line
    theta_, r_ = splev(u_new, tck)

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
        if np.sqrt((base[0] - coord[0])**2 + (base[1] - coord[1])**2) > calibration:
            equal_spaced_points.append(coord)
            base = coord

    equal_spaced_points = np.array(equal_spaced_points)

    if not diagnostics:
        return equal_spaced_points.T
    else:
        return equal_spaced_points.T, x_spline, y_spline, image, tck, u


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

    for f in files:
        part = pickle.load(open(f, 'rb'))
        x1 = part.loc[0]
        x2 = part.loc[0]+part.shape[2]
        y1 = part.loc[1]
        y2 = part.loc[1]+part.shape[3]
        print(f'Index: [{x1}:{x2}, {y1}:{y2}]')
        mask[:, :, x1:x2, y1:y2, :] = part.segmented_mask.astype(part.dtype)

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

    plt.savefig('test.png',dp=1000)
    plt.show()


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

    plt.savefig('test.png',dp=1000)
    plt.show()


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
