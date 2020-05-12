import torch
import numpy as np
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage
import skimage.feature
import scipy.ndimage
import scipy.ndimage.morphology
import transforms as t
from scipy.interpolate import splprep, splev
import pickle
import glob
import ray
from multiprocessing import Pool
import mask

def pad_image_with_reflections(image, pad_size=(30, 30, 6)):
    """
    Pads image according to Unet spe
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

    image_size = image.shape  # expect x,y,z,c
    pad_size = np.array(pad_size)

    left_pad = image.numpy()[pad_size[0]-1::-1, :, :, :]
    left_pad = torch.as_tensor(left_pad.copy())
    right_pad = image.numpy()[-1:-pad_size[0]-1:-1, :, :, :]
    right_pad = torch.as_tensor(right_pad.copy())
    image = torch.cat((left_pad, image, right_pad), dim=0)
    left_pad = 0
    right_pad = 0

    bottom_pad = image.numpy()[:, pad_size[1]-1::-1, :, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, -1:-pad_size[1]-1:-1, :, :]
    top_pad = torch.as_tensor(top_pad.copy())
    image = torch.cat((bottom_pad, image, top_pad), dim=1)
    bottom_pad = 0
    top_pad = 0

    bottom_pad = image.numpy()[ :, :, pad_size[2]-1::-1, :]
    bottom_pad = torch.as_tensor(bottom_pad.copy())
    top_pad = image.numpy()[:, :, -1:-pad_size[2]-1:-1, :]
    top_pad = torch.as_tensor(top_pad.copy())

    return torch.cat((bottom_pad, image, top_pad), dim=2)


def predict_mask(model, image, device):
    """
    Takes in a model and an image and applies the model to all parts of the image.

    ALGORITHM:
    Remove inf and nan ->
    apply padding ->
    calculate indexes for unet based on image.shape ->
    apply Unet on slices of image based on indexes ->
    take only valid portion of the middle of each output mask ->
    construct full valid mask ->
    RETURN mask

    :param model: Trained Unet Model from unet.py
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :param device: 'cuda' or 'cpu'
    :return mask:

    """
    PAD_SIZE = (76, 76, 6)
    EVAL_IMAGE_SIZE = (500, 500, 20)

    mask = torch.zeros((1, 1, image.shape[0], image.shape[1], image.shape[2]), dtype=torch.bool)
    im_shape = image.shape

    # inf and nan screw up model evaluation. Happens occasionally
    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 1

    # Apply Padding
    image = pad_image_with_reflections(torch.as_tensor(image), pad_size=PAD_SIZE)

    #  We now calculate the indicies for our image
    x_ind = calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], image.shape[0])
    y_ind = calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], image.shape[1])
    z_ind = calculate_indexes(PAD_SIZE[2], EVAL_IMAGE_SIZE[2], im_shape[2], image.shape[2])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind) * len(z_ind))-1

    # Loop and apply unet
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            for k, z in enumerate(z_ind):
                print(f'\r{iterations}/{max_iter} ', end=' ')
                # t.to_tensor reshapes image to [B C X Y Z]!!!
                padded_image_slice = t.to_tensor()(image[x[0]:x[1], y[0]:y[1]:, z[0]:z[1], :].numpy()).float()

                # Occasionally everything is just -1 in the whole mat. Skip for speed
                if (padded_image_slice.float() != -1).sum() == 0:
                    iterations += 1
                    continue

                with torch.no_grad():
                    valid_out = model(padded_image_slice.float().to(device))

                valid_out = valid_out[:,:,
                                     PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                                     PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                                     PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2]]


                #do everthing in place to save memory
                # Do the sigmoid in place manually
                # 1/ (1+exp(-x))
                valid_out.mul_(-1)
                valid_out.exp_()
                valid_out.add_(1)
                valid_out.pow_(-1)

                valid_out.gt_(.5)  # Greater Than

                mask[:, :, x[0]:x[0]+valid_out.shape[2],
                           y[0]:y[0]+valid_out.shape[3],
                           z[0]:z[0]+valid_out.shape[4]] = valid_out

                iterations += 1

    print('\ndone!')
    return mask


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

    ind_list = torch.arange(pad_size, image_shape, eval_image_size)
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1]) - pad_size
        z2 = int(z-1) + pad_size
        ind.append([z1, z2])
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([padded_image_shape - (eval_image_size+pad_size * 2), padded_image_shape])
    else:
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

    mask = np.ones((1, 1, x_max, y_max, part.shape[-1]), dtype=np.uint8)

    for f in files:
        part = pickle.load(open(f, 'rb'))
        x1 = part.loc[0]
        x2 = part.loc[0]+part.shape[2]
        y1 = part.loc[1]
        y2 = part.loc[1]+part.shape[3]
        mask[:, :, x1:x2, y1:y2, :] = part.mask.astype(np.uint8)

    return mask


def segment_mask(mask):
    ray.init()
    PAD_SIZE = (10, 10, 0)
    EVAL_IMAGE_SIZE = (1000, 1000, mask.shape[-1])

    im_shape = mask.shape[2::]
    x_ind = calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], mask.shape[2])
    y_ind = calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], mask.shape[3])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind))-1
    distance = np.zeros(mask.shape, dtype=np.half)

    @ray.remote
    def par_fun(x, y, PAD_SIZE, EVAL_IMAGE_SIZE, part):
        distance_part = scipy.ndimage.morphology.distance_transform_edt(mask_slice)

        distance_part = distance_part[:, :,
                        PAD_SIZE[0]:EVAL_IMAGE_SIZE[0] + PAD_SIZE[0],
                        PAD_SIZE[1]:EVAL_IMAGE_SIZE[1] + PAD_SIZE[1],
                        :]
        return x, y, distance_part.astype(np.half)

    # Loop and apply unet
    distance_part_list=[]
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            print(f'\r{iterations}/{max_iter} ', end=' ')
            mask_slice = mask[:, :, x[0]:x[1], y[0]:y[1]:, :]
            distance_part_list.append(par_fun.remote(x, y, PAD_SIZE, EVAL_IMAGE_SIZE, mask_slice))
            iterations += 1

    distance_part_list = ray.get(distance_part_list)

    while distance_part_list:
        part = distance_part_list.pop(0)
        x = part[0]
        y = part[1]

        distance[:,
                 :,
                 x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                 y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                 :] = part[2]

    return distance
