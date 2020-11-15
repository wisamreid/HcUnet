from hcat.r_unet import RecursiveUnet as RUnet, RDCNet
import matplotlib.pyplot as plt
from hcat.unet import Unet_Constructor as Unet
from hcat import mask, utils, rcnn, transforms as t, segment
from hcat.loss import dice, cross_entropy
import hcat
import hcat.dataloader as dataloader
import torch.nn.functional as F
import torch
import sys


# runet = RUnet()
# runet.load('/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/tests/3.3k_epochs_holy_shit.runet')
# runet = runet.cuda()
# runet = runet.train()
# device = 'cuda'

runet = RDCNet(4,5)
runet = runet.cuda()
runet = runet.train()
device = 'cuda'

optimizer = torch.optim.Adam(runet.parameters(), lr = 0.001)

data = dataloader.RecursiveStack(path='../Data/train',
                                        joint_transforms=[t.to_float(),
                                                          t.reshape(),
                                                          # t.nul_crop(),
                                                          # t.random_crop([128, 128, 20]),
                                                          ],
                                        image_transforms=[
                                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                                          ]
                                        )
out_str = ''
for epoch in range(100):

    image, mask, pwl, com, vector = data[0]
    image = image.cuda()
    mask = mask.cuda()
    image = image[:,:,:,:,:-1:]

    if epoch > 0:
        out_str = f'EPOCH: {epoch} | LossProb: {loss_prob.item()}, LossCenter: {"""loss_center.item()"""}, LossVec: {loss_vec.item()}'
        print(out_str, end='\r', flush=True)

    optimizer.zero_grad()
    out = runet(image.float())

    loss_prob = hcat.loss.cross_entropy(out[:,0,:,:,:].unsqueeze(1), mask.to('cuda'), pwl.to('cuda'), method='pixel')
    # loss_center = hcat.loss.L1Loss(out[:,1,:,:,:].unsqueeze(1), com.to('cuda')) * 8
    loss_vec = hcat.loss.MSELoss(out[:,2::,:,:,:], vector.float().to('cuda'))
    loss = loss_prob + loss_vec#loss_center + loss_vec
    loss.backward()
    optimizer.step()

# runet.save('test_2.runet')

with torch.no_grad():
    image, mask, pwl, com, vector  = data[0]
    out = runet(image.float().cuda())
print(out[0,2::, : ,:, :].shape)
labels = segment.pixel_vec_to_cell(out[:, 2::, :, :, :,].cpu().float(), out[0,0,:,:,:].cpu().float() > 0.5)

for i in range(labels.shape[-1]):
    plt.imshow(labels[:,:,i])
    plt.show()
#
#
for i in range(out.shape[-1]):
    fig, ax = plt.subplots(5, 2, figsize=(20, 10))

    plt.tight_layout()

    ax[0,0].imshow(mask[0,0,:,:,i].cpu().float().numpy())
    ax[0,1].imshow(F.sigmoid(out[0,0,:,:,i]).detach().cpu().float().numpy() > 0.5)

    ax[1,0].imshow(com[0,0,:,:,i].cpu().float().numpy())
    ax[1,1].imshow(out[0,1,:,:,i].detach().cpu().float().numpy())

    ax[2,0].imshow(vector[0,0,:,:,i].cpu().float().numpy())
    ax[2,1].imshow(out[0,2,:,:,i].detach().cpu().float().numpy())

    ax[3,0].imshow(vector[0,1,:,:,i].cpu().float().numpy())
    ax[3,1].imshow(out[0,3,:,:,i].detach().cpu().float().numpy())

    ax[4,0].imshow(vector[0,2,:,:,i].cpu().float().numpy())
    ax[4,1].imshow(out[0,4,:,:,i].detach().cpu().float().numpy())

    plt.show()
#
#
#

