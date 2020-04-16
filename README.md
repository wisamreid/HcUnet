# Hair Cell Unet (HcUnet)
######A library of deep learning functions for analysis of confocal z-stacks of mouse cochlear hair cells written in pytorch.

###Quickstart Guide
Bare minimum code requirement for evaluation of an image. 
```python
from unet import unet_constructor as Unet
import torch.nn as nn
import dataloader
import transforms as t


data = dataloader.stack(path='./Data',
                        joint_transforms=[t.to_float(), t.reshape()],
                        image_transforms=[t.normalize()],
                        )

model = Unet(conv_functions=(nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.BatchNorm3d),
             in_channels=4,
             out_channels=1,
             feature_sizes=[8,16,32,64,128],
             kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
             upsample_kernel=(2, 2, 2),
             max_pool_kernel=(2, 2, 1),
             upsample_stride=(2, 2, 1),
             dilation=1,
             groups=1).to('cpu')



image, mask, pwl = data[0]

out = model.forward(image.float())
```




## unet.py
#### Deep Learning Model Defininitions
_class_ **unet.unet_constructor**
```python
unet_constructor(conv_functions=(nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm3d),
                 in_channels=3,
                 out_channels=2,
                 feature_sizes=[32, 64, 128, 256, 512, 1024],
                 kernel=(3, 3),
                 upsample_kernel=(2, 2),
                 max_pool_kernel=(2, 2),
                 upsample_stride=2,
                 dilation=1,
                 groups=1,
                 )

```

