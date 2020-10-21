import torch
import torch.nn as nn
import glob
from warnings import filterwarnings

# try:
#     from hcat.utils import pad_image_with_reflections
# except ModuleNotFoundError:
#     from HcUnet.utils import pad_image_with_reflections


filterwarnings("ignore", category=UserWarning)

class RecursiveUnet(nn.Module):
    """
    Unet will have these sizes: 8, 16, 32
                 kernel_1 =(3, 3, 1),
                 kernel_2 = (3, 3, 1),
                 upsample_kernel=(7, 7, 2),
                 max_pool_kernel=(2, 2),
                 max_pool_kernel=(2, 2, 1),
                 upsample_stride=2,
                 dilation=1,
                 groups=1,
    """
    def __init__(self,
                 image_dimensions=2,
                 in_channels=4,
                 out_channels=5,
                 kernel={'conv1':(3, 3, 2), 'conv2':(3,3,1)},
                 upsample_kernel=(3, 3, 2),
                 max_pool_kernel=(2, 2, 1),
                 upsample_stride=(2, 2, 1),
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
        super(RecursiveUnet, self).__init__()

        # Convert to dict of parameters
        #  In order to allow for multiple values passed to the first and second step of each convolution,
        #  we construct a tuple wit values of 'conv1' and 'conv2 denoting the parameter for each step

        if type(kernel) is tuple:
            kernel = {'conv1': kernel, 'conv2': kernel}
        if type(dilation) is int or type(dilation) is tuple:
            dilation = {'conv1': dilation, 'conv2': dilation}
        if type(groups) is int or type(groups) is tuple:
            groups = {'conv1': groups, 'conv2': groups}

        # Create Dict for saving model
        self.model_specification = {
            'image_dimensions': image_dimensions,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel': kernel,
            'upsample_kernel': upsample_kernel,
            'max_pool_kernel': max_pool_kernel,
            'upsample_stride': upsample_stride,
            'dilation': dilation,
            'groups': groups
                                    }

        channels=[32, 64, 128]

        self.down1 = Down(in_channels=4, out_channels=channels[0], kernel=kernel, dilation=dilation, groups=groups)

        # f_z
        self.down2_fz = Down(in_channels=channels[0], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups)
        self.down3_fz = Down(in_channels=channels[1], out_channels=channels[2], kernel=kernel, dilation=dilation, groups=groups)
        self.up1_fz = Up(in_channels=channels[2], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel = upsample_kernel, upsample_stride = upsample_stride)

        # f_h
        self.down2_fh = Down(in_channels=channels[0], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups)
        self.down3_fh = Down(in_channels=channels[1], out_channels=channels[2], kernel=kernel, dilation=dilation, groups=groups)
        self.up1_fh = Up(in_channels=channels[2], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel=upsample_kernel, upsample_stride=upsample_stride)

        self.up2 = Up(in_channels=channels[1], out_channels=channels[0], kernel=kernel, dilation=dilation, groups=groups,
                      upsample_kernel = upsample_kernel, upsample_stride = upsample_stride)

        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool3d(max_pool_kernel)

        self.fz = f(self.down2_fz, self.down3_fz, self.up1_fz, self.max_pool)
        self.fh = f(self.down2_fh, self.down3_fh, self.up1_fh, self.max_pool)

    def forward(self, x):
        outputs = []

        x = self.down1(x)
        a = x.clone()
        x = self.max_pool(x)

        # Recurrent Bit!!!
        for t in range(1):
            h = self.tanh(self.fh(x))
            z = self.sigmoid(self.fz(x))

            if t == 0:
                h_t = torch.ones(z.shape).cuda()

            h_t = (h_t * z) + (-1 * z * h)


        x = self.up2(h_t, a)
        x = self.out_conv(x)

        return x

    def save(self, filename, hyperparameters=None):
        model = {'state_dict': self.state_dict(),
                 'model_specifications': self.model_specification,
                 'hyperparameters': hyperparameters}

        python_files = {}

        python_files_list= glob.glob('./**/*.py', recursive=True)
        for f in glob.glob('./**/*.ipynb', recursive=True):
            python_files_list.append(f)

        for f in python_files_list:
            file = open(f,'r')
            python_files[f] = file.read()
            file.close()

        model['python_files'] = python_files
        model['tree_structure'] = glob.glob('**/*', recursive=True)

        torch.save(model, filename)
        return None

    def load(self, filename, to_cuda=True):

        if torch.cuda.is_available() and to_cuda:
            device = 'cuda:0'
        else:
            Warning('Cuda is not available, initializing model on the CPU')
            device = 'cpu'

        model = torch.load(filename, map_location=device)
        model_specification = model['model_specifications']

        self.__init__()

        self.load_state_dict(model['state_dict'])
        self.eval()
        try:
            return model['hyperparameters']
        except KeyError:
            return None

class f(nn.Module):
    def __init__(self, down1, down2, up1, max_pool):
        super(f, self).__init__()
        self.down1 = down1
        self.down2 = down2
        self.up1 = up1
        self.max_pool = max_pool

    def forward(self, x):
        x = self.down1(x)
        b = x.clone()
        x = self.max_pool(x)
        x = self.down2(x)
        x = self.up1(x, b)
        return x


class Down(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: dict,
                 dilation: dict,
                 groups: dict,
                 ):
        super(Down, self).__init__()

        self.conv1 = nn.Conv3d(in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=0)

        self.conv2 = nn.Conv3d(out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=0)

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.batch1(self.conv1(x)))
        x = self.relu(self.batch2(self.conv2(x)))
        return x


class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 upsample_kernel: tuple,
                 upsample_stride: int,
                 dilation: dict,
                 groups: dict,
                 ):

        super(Up, self).__init__()
        self.conv1 = nn.Conv3d(in_channels,
                                   out_channels,
                                   kernel['conv1'],
                                   dilation=dilation['conv1'],
                                   groups=groups['conv1'],
                                   padding=0)
        self.conv2 = nn.Conv3d(out_channels,
                                   out_channels,
                                   kernel['conv2'],
                                   dilation=dilation['conv2'],
                                   groups=groups['conv2'],
                                   padding=0)

        self.up_conv = nn.ConvTranspose3d(in_channels,
                                         out_channels,
                                         upsample_kernel,
                                         stride=upsample_stride,
                                         padding=0)
        self.lin_up = False

        self.batch1 = nn.BatchNorm3d(out_channels)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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