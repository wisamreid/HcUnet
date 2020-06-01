import torch
import torch.nn as nn


def cross_entropy_loss(pred, mask, pwl, weight='worst_z'):
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



    # if mask.long().max() == 1:
    #
    # else:
    #     cel = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor([0,2.25]).float().cuda())
    #     l = cel(pred, mask)

    #cel = nn.BCEWithLogitsLoss(reduction='none', weight=torch.tensor([weight]).float().cuda())
    cel = nn.BCEWithLogitsLoss(reduction='none')
    l = cel(pred.float(), mask.float())
    loss = (l*(pwl+1))
    if weight == 'worst_z':
        scaling = torch.linspace(1, 2, pred.shape[4]) ** 2
        loss, _ = torch.sort(loss.sum(dim=[0,1,2,3]))
        loss *= scaling.to(loss.device)
        loss /= (pred.shape[2]*pred.shape[3])

    return loss.mean()


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
    #intersection = pred * mask
    #union = (pred + mask).sum()

    loss = (2*(pred * mask).sum()+1e-10) / ((pred + mask).sum() + 1e-10)

    return 1-loss


def random_cross_entropy(pred, mask, pwl, weight, size):
    pred_shape = pred.shape
    if len(pred_shape) == 5:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1, 0:pred_shape[4]:1]
    elif len(pred_shape) == 4:
        mask = mask[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
        pwl = pwl[:, :, 0:pred_shape[2]:1, 0:pred_shape[3]:1]
    else:
        raise IndexError(f'Unexpected number of predicted mask dimmnsions. Expected 4 (2D) or 5 (3D) but got' +
                         f' {len(pred_shape)} dimensions: {pred_shape}')
    if size <= 1:
        raise ValueError(f'Size should be greater than 1 not {size}')
    if (mask==1).sum() == 0:
        raise ValueError(f'No positive labels in the mask')
    if (mask==0).sum() == 0:
        raise ValueError(f'No negative labels in the mask')

    cel = nn.BCEWithLogitsLoss(reduction='none')

    pred = pred.reshape(-1)
    mask = mask.reshape(-1)
    pwl = pwl.reshape(-1)

    pos_ind = torch.randint(low=0, high=int((mask==1).sum()), size=(1, size))[0, :]
    neg_ind = torch.randint(low=0, high=int((mask==0).sum()), size=(1, size))[0, :]

    pred = torch.cat([pred[mask==1][pos_ind], pred[mask==0][neg_ind]]).unsqueeze(0)
    pwl = torch.cat([pwl[mask==1][pos_ind], pwl[mask==0][neg_ind]]).unsqueeze(0)
    mask = torch.cat([mask[mask==1][pos_ind], mask[mask==0][neg_ind]]).unsqueeze(0)

    l = cel(pred.float(), mask.float())

    return (l*(pwl+1)).mean()
