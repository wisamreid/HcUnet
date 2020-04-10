import torch
import torch.nn as nn



def cross_entropy_loss(pred, mask, pwl):
    pred_shape = pred.shape
    if len(pred_shape) == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif len(pred_shape) == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimmensions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {len(pred_shape)} dimmensions: {pred_shape}')

    cel = nn.CrossEntropyLoss(reduction='none')
    cel = nn.CrossEntropyLoss(reduction='none')

    # loss = cel(pred.squeeze(0).reshape(2,-1).transpose(1,0), mask.long().reshape(-1))
    # loss = torch.sum(loss * 3 * pwl.reshape(-1))
    loss = cel(pred, mask.long().squeeze(1))  # HAVE TO SQUEEZE FEATURE DIM IN THIS CASE FOR SOME REASON.

    return (loss * (pwl ** 2)).sum()


def dice_loss(pred, mask):
    pred_shape = pred.shape
    if len(pred_shape) == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif len(pred_shape) == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimmensions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {len(pred_shape)} dimmensions: {pred_shape}')

    intersection = pred * mask
    union = (pred ** 2).sum() + (mask ** 2).sum()

    loss = (2 * intersection.sum()) / union

    return (1 / (loss + 1e-10)) ** 2
