from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import time


def train(model, dataloader, optimizer, scheduler, num_epochs, lr, gamma, device, silent=False):

    print(f'Started training at: {time.asctime(time.localtime())}')
    l = len(dataloader)
    losses = []
    summed_losses = []
    previous_average_loss = 0
    previous_summed_loss = 0
    start_time = 0

    for e in range(num_epochs):
        summed_loss = 0
        k = 1
        for ind, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            for i in labels:
                labels[i] = labels[i].to(device)

            if torch.isnan(images).sum() > 0:
                raise ValueError('image is nan')
            if torch.isinf(images).sum() > 0:
                raise ValueError('image is inf')

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

        if e % 1 == 0 and not silent:
            print(f'\rEpoch ' + ' ' * (4 - len(str(e))) + f'\033[31m{e}\033[0m |', end='')
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
        scheduler.step()
        lr *= gamma

    print(f'\nFinished at: {time.asctime(time.localtime())}')

    return model, summed_losses




