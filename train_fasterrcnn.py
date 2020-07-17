from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dataloader
import matplotlib.pyplot as plt
import utils as u
import transforms as t
import time
import skimage.exposure
# I think we need to have 5 classes, where 0 is background...
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                             progress=True,
                                                             num_classes=5,
                                                             pretrained_backbone=True,
                                                             box_detections_per_img=500)

model.load_state_dict(torch.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Jun26_18:12.pth'))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.train()
model = model.to(device)

norm = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

data = dataloader.section(path='./Data/FasterRCNN_trainData/Top',
                          image_transforms=[t.to_float(),
                                            t.random_gamma((.8, 1.2)),
                                            t.random_intensity(),
                                            t.spekle(0.00001),
                                            t.remove_channel(remaining_channel_index=[0, 2, 3]),
                                            t.normalize(mean=norm['mean'], std=norm['std']),
                                            ],
                          joint_transforms=[
                                            t.random_x_flip(),
                                            t.random_y_flip(),
                                            t.add_junk_image(path='Data/FasterRCNN_junkData/',
                                                             junk_image_size=(100, 100),
                                                             normalize=norm)
                                            # t.random_resize(scale=(.3, 4)),
                                            ]
                          )

# Hyper Parameters
num_epochs = 2000
lr = 1e-6
gamma =  0.98

# Random Initializations
l = len(data)
losses = []
summed_losses = []
epoch = 0
loss_labels = ['LC', 'LBR', 'LO', 'LRBR']
previous_average_loss = 0
previous_summed_loss = 0


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.965)

TRAIN = True

print(f'Started at: {time.asctime(time.localtime())}')
if TRAIN:
    start_time = 0
    for e in range(num_epochs):
        summed_loss = 0
        k = 1
        for ind, (images, labels) in enumerate(data):
            images = images.to(device)
            for i in labels:
                labels[i] = labels[i].to(device)

            if torch.isnan(images).sum() > 0:
                raise ValueError('image is nan')
            if torch.isinf(images).sum() > 0:
                raise ValueError('image is inf')

            elapsed_time = time.perf_counter() - start_time

            optimizer.zero_grad()
            
            outputs = model(images.float(), [labels])

            loss = None
            for i in outputs:
                if loss is None:
                    loss = outputs[i]
                else:
                    loss += outputs[i]

            summed_loss += loss.item()
            losses.append(loss.item())
            k += 1

            loss.backward()
            optimizer.step()
            # Make this shit pretty

        if e % 1 == 0:
            print(f'\rEpoch ' + ' ' * (4 - len(str(e))) + f'\033[31m{e}\033[0m |', end='')
            #    print('['+ '█' * k +' '*(l-k)  + ']',end='')
            #    print(' '+ ' '*(2-len(str(k))) +'\033[33;1m' + str(k) + f'\033[0m/\033[32;1m{l}\033[0m |',end='')
            #        for i,lab in enumerate(outputs):
            #            print(' \033[35m'+loss_labels[i]+' \033[0m' + str(outputs[lab].item())[0:8] ,end='')
            print(f' \033[35mPAL: \033[0m{str(previous_average_loss)[0:8]}', end='')
            print(f' \033[36mAL: \033[0m{str(summed_loss / k)[0:8]}', end='')
            print(f' \033[35mPSL: \033[0m{str(previous_summed_loss)[0:8]}', end='')
            print(f' \033[36mSL: \033[0m{str(summed_loss)[0:8]} \033[0m|', end='')
            print(f' \033[34mTE: \033[0m{str((time.perf_counter() - start_time))[0:8]} sec', end='')
            print(f' \033[34mLR: \033[0m{str(lr)}', end='')

            previous_summed_loss = np.copy(summed_loss)
            previous_average_loss = np.copy(summed_loss / l)
            start_time = time.perf_counter()

        summed_losses.append(summed_loss)

    print(f'\nFinished at: {time.asctime(time.localtime())}')


    torch.save(model.state_dict(), 'best_1fasterrcnn.pth')

images, _ = data[19]
model.eval()
with torch.no_grad():
    a = model(images.to(device).float())

u.show_box_pred(images.squeeze().float(), a, .4)

plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.show()

