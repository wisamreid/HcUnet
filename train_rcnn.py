from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torchvision
from hcat import dataloader, utils as u, transforms as t
import hcat.train
import matplotlib.pyplot as plt
import time

# I think we need to have 5 classes, where 0 is background...
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                             progress=True,
                                                             num_classes=3,
                                                             pretrained_backbone=True,
                                                             box_detections_per_img=500)
model.load_state_dict(torch.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Aug18_16:02.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.train()
model = model.to(device)

norm = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

data = dataloader.Section(path='./Data/FasterRCNN_trainData/Top/',
                          simple_class=True,
                          image_transforms=[t.to_float(),
                                            t.random_gamma((.85, 1.15)),
                                            t.random_intensity(),
                                            t.spekle(0.00001),
                                            t.remove_channel(remaining_channel_index=[0, 2, 3]),
                                            t.normalize(**norm),
                                            ],
                          joint_transforms=[
                                            t.random_x_flip(),
                                            t.random_y_flip(),
                                            t.add_junk_image(path='./Data/FasterRCNN_junkData/',
                                                             junk_image_size=(100, 100),
                                                             normalize=norm),
                                            t.add_junk_image(path='./Data/FasterRCNN_junkData/',
                                                             junk_image_size=(100, 100),
                                                             normalize=norm)
                                            # t.random_resize(scale=(.3, 4)),
                                            ]
                          )



# Hyper Parameters
num_epochs = 200
lr = 1e-5
gamma = 0.998
scale = 3

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

model, summed_losses = hcat.train.frcnn(model, data, optimizer, scheduler, num_epochs, lr, gamma, device, scale=2)
torch.save(model.state_dict(), 'fasterrcnn_' + time.strftime('%b%d_%H:%M') + '.pth')
print(f'Saved: {"fasterrcnn_" + time.strftime("%b%d_%H:%M") + ".pth"}')
images, _ = data[19]
model.eval()
with torch.no_grad():
    a = model(images.to(device).float())

u.show_box_pred(images.squeeze().float(), a, .3)

plt.figure()
plt.plot(summed_losses)
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.show()

