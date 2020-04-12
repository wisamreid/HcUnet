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

    if mask.long().max() == 1:
        cel = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([2]).float().cuda())
        l = cel(pred.float(), mask.float())

    else:
        cel = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([0,2.25]).float().cuda())
        l = cel(pred, mask)


    # loss = cel(pred, mask.long().squeeze(1))  # HAVE TO SQUEEZE FEATURE DIM IN THIS CASE FOR SOME REASON.

    return (l*pwl).mean()


def dice_loss(pred, mask):
    pred_shape = pred.shape
    if len(pred_shape) == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif len(pred_shape) == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimmensions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {len(pred_shape)} dimmensions: {pred_shape}')

    pred = torch.sigmoid(pred)
    intersection = pred * mask
    union = (pred + mask).sum()

    loss = (2*intersection.sum()+1e-10 ) / (union+1e-10)

    return 1-loss

