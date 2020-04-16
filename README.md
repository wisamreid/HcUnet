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

###_class_ **unet_constructor**
```python
model = unet_constructor(conv_functions=(nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm3d),
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
#### _fun_ forward
```python
model.forward(image: torch.Tensor(dtype=torch.float))
```
*image*: torch.Tensor of type _float_ with shape [B, C, X, Y, Z] 
* B: Batch Size
* C: Number of Channels as defined by variable _in_channels_
* X: Size of image in x dimension
* Y: Size of image in y dimension
* Z: Size of image in z dimension

*returns*: ouput mask of torch.Tensor of type _float_ with shape [B, M, X*, Y*, Z*]
* B: Batch Size
* M: Number of mask Channels as defined by variable _out_channels_
* X*: Size of mask in x dimension
* Y*: Size of mask in y dimension
* Z*: Size of mask in z dimension

In all cases X*, Y*, and Z* will be less than X, Y, and Z due to only valid convolutions being used in the forward pass. The output mask will contain gradients unless model.eval() is called first. 

#### _fun_ save
```python
model.save(filename: str)
```
*filename*: filename by which to serialize model state to

This function serializes the state of the model as well as initialization parameters. The save model can be loaded with model.load(filename)

#### _fun_ load
```python
model.load(filename: str, to_cuda=True)
```
*filename* Filename which to load model from. 
*to_cuda* If true attemts to load the model state to cuda. If cuda is not available will throw a warning and initalize on the cpu instead. 