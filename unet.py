import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from utils import pad_image_with_reflections
except ModuleNotFoundError:
    from HcUnet.utils import pad_image_with_reflections


class unet_constructor(nn.Module):
    def __init__(self, conv_functions=(nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm3d),
                 in_channels=3,
                 out_channels=2,
                 feature_sizes=[32, 64, 128, 256, 512, 1024],
                 kernel=(3, 3),
                 upsample_kernel=(2, 2),
                 max_pool_kernel=(2, 2),
                 upsample_stride=2,
                 dilation=1,
                 groups=1,
                 ):
        """
        A Generic Unet Builder Allowing for various different symetric archetectures.
        Generates lists of functions which are looped over in self.forward() to evaluate Unet

        Can pass Dict for each


        :param conv_functions: Tuple of functions from torch.nn
            conv_functions[0]: torch.nn.Conv2d or torch.nn.Conv3d
            conv_functions[1]: torch.nn.ConvTranspose2d or torch.nn.ConvTranspose3d
            conv_functions[2]: torch.nn.MaxPool2d or torch.nn.MaxPool3d

        :param in_channels: Int: Number of input Color Channels; Default is 3
        :param out_channels: Int: Number of classes to predict masks for
        :param kernel: Tuple: Tuple: Convolution Kernel for function in conv_functions[0]
        :param upsample_kernel: Tuple: Convolution Kernel for torch.nn.ConvTranspose2/3d in conv_functions[1]
        :param max_pool_kernel: Tuple: Kernel for torch.nn.MaxPool2/3d
        :param feature_sizes: List: List of integers describing the number of feature channels at each step of the U
        """
        super(unet_constructor, self).__init__()


        # Convert to dict of parameters
        #  In order to allow for multiple values passed to the first and second step of each convolution,
        #  we construct a tuple wit values of 'conv1' and 'conv2 denoting the parameter for each step

        if type(kernel) is tuple:
            kernel = {'conv1': kernel, 'conv2': kernel}
        if type(dilation) is int or type(dilation) is tuple:
            dilation = {'conv1': dilation, 'conv2': dilation}
        if type(groups) is int or type(groups) is tuple:
            groups = {'conv1': groups, 'conv2': groups}

        # Error Checking
        if len(feature_sizes) < 2:
            raise ValueError(f'The Number of Features must be at least 2, not {len(feature_sizes)}')
        for i, f in enumerate(feature_sizes[0:-1:1]):
            assert f*2 == feature_sizes[i+1], \
                f'Feature Sizes must be multiples of two from each other: {f} != {feature_sizes[i-1]}*2'

        # Create Dict for saving model
        self.model_specification = {
            'conv_functions': conv_functions,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feature_sizes': feature_sizes,
            'kernel': kernel,
            'upsample_kernel': upsample_kernel,
            'max_pool_kernel': max_pool_kernel,
            'upsample_stride': upsample_stride,
            'dilation': dilation,
            'groups':groups
                                    }

        self.down_steps = []
        self.up_steps = []

        self.down_steps.append(Down(conv_functions,
                                    in_channels=in_channels,
                                    out_channels=feature_sizes[0],
                                    kernel=kernel,
                                    dilation=dilation,
                                    groups=groups,
                                    ))

        i = 1
        for f in feature_sizes[1::]:
            self.down_steps.append(Down(conv_functions,
                                        in_channels=feature_sizes[i-1],
                                        out_channels=f,
                                        kernel=kernel,
                                        dilation=dilation,
                                        groups=groups,
                                        ))
            i += 1
        i = -2
        for f in feature_sizes[:0:-1]:
            self.up_steps.append(Up(conv_functions,
                                    in_channels=f,
                                    out_channels=feature_sizes[i],
                                    kernel=kernel,
                                    upsample_kernel=upsample_kernel,
                                    upsample_stride=upsample_stride,
                                    dilation=dilation,
                                    groups=groups,
                                    ))
            i += -1

        self.out_conv = conv_functions[0](feature_sizes[0], out_channels, 1)
        self.down_steps = nn.ModuleList(self.down_steps)
        self.up_steps = nn.ModuleList(self.up_steps)
        self.max_pool = conv_functions[2](max_pool_kernel)

    def forward(self, x):
        outputs = []

        for step in self.down_steps[:-1]:
            x = step(x)
            outputs.append(x)
            x = self.max_pool(x)

        x = self.down_steps[-1](x)

        for step in self.up_steps:
            x = step(x, outputs.pop())

        x = self.out_conv(x)

        outputs = None  # Do this to clear memory

        return x



    def save(self):
        model = {'state_dict': self.state_dict(),
                 'model_specifications': self.model_specification}
        torch.save(model, 'model.unet')
        return None

    def load(self, path):

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'

        model = torch.load(path, map_location=device)
        model_specification = model['model_specifications']

        self.__init__(
                 conv_functions=model_specification['conv_functions'],
                 in_channels=model_specification['in_channels'],
                 out_channels=model_specification['out_channels'],
                 feature_sizes=model_specification['feature_sizes'],
                 kernel=model_specification['kernel'],
                 upsample_kernel=model_specification['upsample_kernel'],
                 max_pool_kernel=model_specification['max_pool_kernel'],
                 upsample_stride=model_specification['upsample_stride'],
                 dilation=model_specification['dilation'],
                 groups=model_specification['groups'],
                 )

        self.load_state_dict(model['state_dict'])
        self.eval()
        return None

    def evaluate(self, image: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise ValueError(f'Expected image type of torch.Tensor, not {type(image)}')
        if image.shape[1] != self.model_specification['in_channels']:
            raise ImportError(f'Image expected to have {self.model_specification["in_channels"]} not {image.shape[1]}')

        self.eval()

        pad = (100, 100, 8)
        mask = torch.zeros([image.shape[0],
                            self.model_specification['out_channels'],
                            image.shape[2],
                            image.shape[3],
                            image.shape[4]])

        skip = 200
        trusted_image_size = 100

        padded_image = pad_image_with_reflections(image, pad).float()

        for x in torch.arange(0, padded_image.shape[2], skip):
            print(x)
            for y in torch.arange(0, padded_image.shape[3], skip):
                x = int(x)
                y = int(y)

                try:
                    slice_to_eval = padded_image[:, :, x:x+skip, y:y+skip, : ]
                except IndexError:
                    slice_to_eval = padded_image[:, :, x::, y::, :]

                mask_slice = self.forward(slice_to_eval)[:, :,
                                                         pad[0] // 2:skip,
                                                         pad[1] // 2:skip,
                                                         pad[2] // 2: image.shape[3]:1]



class Down(nn.Module):
    def __init__(self, conv_functions: tuple,
                 in_channels: int,
                 out_channels: int,
                 kernel: dict,
                 dilation: dict,
                 groups: dict,
                 ):
        super(Down, self).__init__()

        self.conv1 = conv_functions[0](in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=0)
        self.conv2 = conv_functions[0](out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=0)

        self.batch1 = conv_functions[3](out_channels)
        self.batch2 = conv_functions[3](out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x

class Up(nn.Module):
    def __init__(self, conv_functions: tuple,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 upsample_kernel: tuple,
                 upsample_stride: int,
                 dilation: dict,
                 groups: dict,
                 ):
        super(Up, self).__init__()
        self.conv1 = conv_functions[0](in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=0)
        self.conv2 = conv_functions[0](out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=0)

        self.up_conv = conv_functions[1](in_channels,
                                         out_channels,
                                         upsample_kernel,
                                         stride=upsample_stride,
                                         padding=0)

        self.batch1 = conv_functions[3](out_channels)
        self.batch2 = conv_functions[3](out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.up_conv(x)
        y = crop(x, y)
        x = torch.cat((x, y), dim=1)
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x


@torch.jit.script
def crop(x, y):
    """
    Cropping Function to crop tensors to each other. By default only crops last 2 (in 2d) or 3 (in 3d) dimensions of
    a tensor.

    :param x: Tensor to be cropped
    :param y: Tensor by who's dimmension will crop x
    :return:
    """
    shape_x = x.shape
    shape_y = y.shape
    cropped_tensor = torch.empty(0)

    assert shape_x[1] == shape_y[1],\
        f'Inputs do not have same number of feature dimmensions: {shape_x} | {shape_y}'

    if len(shape_x) == 4:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1]
    if len(shape_x) == 5:
        cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1, 0:shape_y[4]:1]

    return cropped_tensor




