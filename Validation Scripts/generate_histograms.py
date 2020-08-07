import dataloader as dataloader
import transforms as t

import numpy as np
import matplotlib.pyplot as plt
from hcat.unet import unet_constructor as GUnet
from hcat.loss import dice_loss
import torch

data = dataloader.stack(path='/home/chris/Desktop/ColorImages',
                        joint_transforms=[t.to_float(),
                                          t.reshape(),
                                          t.random_crop([512, 512, 30]),
                                          ],
                        image_transforms=[
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ]
                        )

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print('Initalizing Unet:  ',end='')
unet= GUnet(image_dimensions=3,
            in_channels=4,
            out_channels=1,
            feature_sizes=[16,32,64,128],
            kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
            upsample_kernel=(8, 8, 2),
            max_pool_kernel=(2, 2, 1),
            upsample_stride=(2, 2, 1),
            dilation=1,
            groups=2).to(device)

# unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Jun7_chris-MS-7C37_1.unet')
unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Jun25_chris-MS-7C37_1.unet')
unet.to(device)
unet.eval()
print('Done')


for i, (image, mask, pwl) in enumerate(data):
    print(image.shape)
    with torch.no_grad():
        valid_out = unet(image.cuda().float())

    # valid_out.mul_(-1)
    # valid_out.exp_()
    # valid_out.add_(1)
    # valid_out.pow_(-1)
    # valid_out.gt_(.50)  # Greater Than
    # valid_out = valid_out.type(torch.uint8)

    dl = dice_loss(valid_out, mask.cuda())
    print(dl, end=' ')
    valid_out = valid_out.float()
    valid_out.mul_(-1)
    valid_out.exp_()
    valid_out.add_(1)
    valid_out.pow_(-1)
    valid_out.gt_(.50)  # Greater Than
    out = valid_out.cpu().detach().bool().numpy()

    mask = mask[:, :, 0:out.shape[2]:1, 0:out.shape[3]:1, 0:out.shape[4]:1].float()
    image = image[:, :, 0:out.shape[2]:1, 0:out.shape[3]:1, 0:out.shape[4]:1].float() * 0.5 + 0.5
    out = out > 0
    mask = mask > 0

    ind = np.bitwise_xor(out, mask)
    print(ind.dtype, mask.shape, out.shape)
    print(f'Num Px Missed: {(mask[ind>0] > 0).sum()}, Num Pix Incorrectly Labeled: {(out[ind>0] > 0).sum()}',end=' | ')
    mis =(mask[ind>0] > 0).sum().float()
    ms=mask.sum().float()
    ratio = mis/ms
    print(ratio,  (out[ind>0] > 0).sum() / out.sum())

    plt.figure()
    plt.yscale('log')
    plt.hist(image[0,1,:,:,:][mask[0,0,:,:,:]],bins=50,alpha=0.5)
    plt.hist(image[0,1,:,:,:][out[0,0,:,:,:]],bins=50,alpha=0.5)
    plt.legend(['Manual', 'Automatic'])
    plt.title(data.files[i],fontdict={'fontsize': 8})
    plt.xlabel('GFP')
    plt.show()


# test = test.type(torch.float)
#
# image, mask, pwl = data[0]
#
# out = test.forward(image.float().to(device))
# out_loss = dice_loss(out, mask.to(device))#, pwl.float().to('cuda'))
# out_loss = cross_entropy_loss(out, mask.to(device), pwl.float().to(device), weight='worst_z')
# #
#
#
#













