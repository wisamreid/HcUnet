from hcat.r_unet import RecursiveUnet as RUnet
import matplotlib.pyplot as plt
from hcat.unet import Unet_Constructor as Unet
from hcat import mask, utils, rcnn, transforms as t, segment
from hcat.loss import dice, cross_entropy
import hcat
import hcat.dataloader as dataloader
import torch
import sys


runet = RUnet()
runet = runet.cuda()
runet = runet.train()
device = 'cuda'

optimizer = torch.optim.Adam(runet.parameters(), lr = 0.1)

data = dataloader.RecursiveStack(path='../Data/train',
                                        joint_transforms=[t.to_float(),
                                                          t.reshape(),
                                                          t.random_crop([300, 300, 30]),
                                                          ],
                                        image_transforms=[
                                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                                          ]
                                        )

image, mask, pwl, com, vector  = data[0]

image = image.cuda()
mask = mask.cuda()

out_str = ''
for epoch in range(500):

    if epoch > 0:
        out_str = f'EPOCH: {epoch} | LossProb: {loss_prob.item()}, LossCenter: {loss_center.item()}, LossVec: {loss_vec.item()}'
        print(out_str,end='\r')
    optimizer.zero_grad()
    out = runet(image.float())

    loss_prob = hcat.loss.cross_entropy(out[:,0,:,:,:].unsqueeze(1), mask.to('cuda'), pwl.to('cuda'), method='pixel')
    loss_center = hcat.loss.L1Loss(out[:,1,:,:,:].unsqueeze(1), com.to('cuda')) * 8
    loss_vec = hcat.loss.MSELoss(out[:,2::,:,:,:], vector.float().to('cuda'))
    loss = loss_prob + loss_center + loss_vec
    loss.backward()
    optimizer.step()





out = torch.sigmoid(out) > 0.5

for i in [15]:
    plt.imshow(out[0,0,:,:,i].detach().cpu().float().numpy())
    plt.show()
    plt.imshow(out[0,1,:,:,i].detach().cpu().float().numpy())
    plt.show()
    plt.imshow(out[0,2,:,:,i].detach().cpu().float().numpy())
    plt.show()
    plt.imshow(out[0,3,:,:,i].detach().cpu().float().numpy())
    plt.show()
    plt.imshow(out[0,4,:,:,i].detach().cpu().float().numpy())
    plt.show()


