import torch
import torch.nn as nn


def cross_entropy(pred: torch.Tensor, mask: torch.Tensor, pwl: torch.Tensor, method='pixel', num_random_pixels=None):
    """
    Pytorch Implementation of Cross Entropy Loss with a pixel by pixel weighting as described in U-NET



    :param pred: torch.Tensor | probability map of shape [B,C,X,Y,Z] predicted by hcat.unet
    :param mask: torch.Tensor | ground truth probability map of shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped
                 to identical size of pred
    :param pwl:  torch.Tensor | weighting map pf shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped to be identical in
                 size to pred and will be used to multiply the cross entropy loss at each pixel.
    :param method: ['pixel', 'worst_z'] | method by which to weight loss
                        - pixel: weights each pixel size by values in pwl
                        - worst_z: adds additional exponentially decaying weight to z planes on with the largest weight
                                   to the worst performing z plane
                        - random: randomly chooses
    :param num_random_pixels: int or None | Number of randomly selected pixels to draw when method='random'
    :return: torch.float | Average cross entropy loss of all pixels of the predicted mask vs ground truth
    """

    _methods = ['pixel', 'worst_z', 'random']
    if method not in _methods:
        raise ValueError(f'Viable methods for cross entropy loss are {_methods}, not {method}.')

    if method == 'pixel':
        if num_random_pixels is None:
            raise ValueError(f'the number of random pixels to draw is not defined. Please set num_random_pixels to a ' +
                             f'value larger than 1.')
        if num_random_pixels <= 1:
            raise ValueError(f'num_random_pixels should be greater than 1 not {num_random_pixels}.')
        if (mask == 0).sum() == 0:
            raise ValueError(f'There are no background pixels in mask.\n\t(mask==0).sum() == 0 -> True')

    pred_shape = pred.shape
    n_dim = len(pred_shape)

    # Crop mask and pwl to the same size as pred
    if n_dim == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif n_dim == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimensions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {n_dim} dimensions: {pred_shape}')

    # Hacky way to do this:
    pwl[mask > .5] += 2

    cel = nn.BCEWithLogitsLoss(reduction='none')
    loss = None

    if method == 'pixel':
        loss = cel(pred.float(), mask.float())
        loss = (loss * (pwl + 1))

    elif method == 'worst_z':
        loss = cel(pred.float(), mask.float())
        loss = (loss * (pwl + 1))
        scaling = torch.linspace(1, 2, pred.shape[4]) ** 2
        loss, _ = torch.sort(loss.sum(dim=[0, 1, 2, 3]))
        loss *= scaling.to(loss.device)
        loss /= (pred.shape[2]*pred.shape[3])

    elif method == 'random':
        pred = pred.reshape(-1)
        mask = mask.reshape(-1)

        if (mask == 1).sum() == 0:
            loss = cel(pred.float(), mask.float())
        else:
            pos_ind = torch.randint(low=0, high=int((mask == 1).sum()), size=(1, num_random_pixels))[0, :]
            neg_ind = torch.randint(low=0, high=int((mask == 0).sum()), size=(1, num_random_pixels))[0, :]

            pred = torch.cat([pred[mask == 1][pos_ind], pred[mask == 0][neg_ind]]).unsqueeze(0)
            mask = torch.cat([mask[mask == 1][pos_ind], mask[mask == 0][neg_ind]]).unsqueeze(0)

            loss = cel(pred.float(), mask.float())

    return loss.mean()


def dice(pred: torch.Tensor, mask: torch.Tensor):
    """
    Calculates the dice loss between pred and mask

    :param pred: torch.Tensor | probability map of shape [B,C,X,Y,Z] predicted by hcat.unet
    :param mask: torch.Tensor | ground truth probability map of shape [B, C, X+dx, Y+dy, Z+dz] that will be cropped
                 to identical size of pred
    :return: torch.float | calculated dice loss
    """

    pred_shape = pred.shape
    n_dim = len(pred_shape)

    if n_dim == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif n_dim == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimensions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {n_dim} dimensions: {pred_shape}')

    pred = torch.sigmoid(pred)
    loss = (2 * (pred * mask).sum() + 1e-10) / ((pred + mask).sum() + 1e-10)

    return 1-loss


# def random_cross_entropy(pred, mask, pwl, size):
#     raise DeprecationWarning
#
#     pred_shape = pred.shape
#     n_dim = len(pred_shape)
#
#     if n_dim == 5:
#         mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
#         pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
#     elif n_dim == 4:
#         mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
#         pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
#     else:
#         raise IndexError(f'Unexpected number of predicted mask dimensions. Expected 4 (2D) or 5 (3D) but got' +
#                          f' {len(pred_shape)} dimensions: {pred_shape}')
#
#     if size <= 1:
#         raise ValueError(f'Size should be greater than 1 not {size}')
#
#     if (mask == 1).sum() < size:
#         size = (mask == 1).sum()
#
#     if (mask==0).sum() == 0:
#         raise ValueError(f'No negative labels in the mask')
#
#     elif (mask==1).sum() == 0:
#         cel = nn.BCEWithLogitsLoss(reduction='none')
#         pred = pred.reshape(-1)
#         mask = mask.reshape(-1)
#         pwl = pwl.reshape(-1)
#         l = cel(pred.float(), mask.float())
#
#     else:
#         cel = nn.BCEWithLogitsLoss(reduction='none')
#
#         pred = pred.reshape(-1)
#         mask = mask.reshape(-1)
#         pwl = pwl.reshape(-1)
#
#         pos_ind = torch.randint(low=0, high=int((mask==1).sum()), size=(1, size))[0, :]
#         neg_ind = torch.randint(low=0, high=int((mask==0).sum()), size=(1, size))[0, :]
#
#         pred = torch.cat([pred[mask==1][pos_ind], pred[mask==0][neg_ind]]).unsqueeze(0)
#         pwl = torch.cat([pwl[mask==1][pos_ind], pwl[mask==0][neg_ind]]).unsqueeze(0)
#         mask = torch.cat([mask[mask==1][pos_ind], mask[mask==0][neg_ind]]).unsqueeze(0)
#
#         l = cel(pred.float(), mask.float())
#
#     return (l*(pwl+1)).mean()
