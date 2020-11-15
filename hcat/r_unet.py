import torch
import torch.nn as nn
import glob
from warnings import filterwarnings

# try:
#     from hcat.utils import pad_image_with_reflections
# except ModuleNotFoundError:
#     from HcUnet.utils import pad_image_with_reflections


filterwarnings("ignore", category=UserWarning)

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
                 kernel={'conv1':(3, 3, 3), 'conv2':(3, 3, 3)},
                 upsample_kernel=(6, 6, 5),
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

        channels=[16, 32, 64]

        # In channels: [R G B Y ProbMap, Centers, X_vec, Y_vec, Z_vec]
        self.down1 = Down(in_channels=9, out_channels=channels[0],
                          kernel=kernel, dilation=dilation, groups=groups, padding=1)

        # f_z
        self.down2_fz = Down(in_channels=channels[0], out_channels=channels[1],
                             kernel=kernel, dilation=dilation, groups=groups,padding=1)
        self.down3_fz = Down(in_channels=channels[1], out_channels=channels[2],
                             kernel=kernel, dilation=dilation, groups=groups,padding=1)
        self.up1_fz = Up(in_channels=channels[2], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel = upsample_kernel, upsample_stride = upsample_stride, padding_down=1,padding_up=2)

        # f_h
        self.down2_fh = Down(in_channels=channels[0], out_channels=channels[1],
                             kernel=kernel, dilation=dilation, groups=groups, padding=1)
        self.down3_fh = Down(in_channels=channels[1], out_channels=channels[2],
                             kernel=kernel, dilation=dilation, groups=groups, padding=1)
        self.up1_fh = Up(in_channels=channels[2], out_channels=channels[1], kernel=kernel, dilation=dilation, groups=groups,
                         upsample_kernel=upsample_kernel, upsample_stride=upsample_stride, padding_down=1, padding_up=2)

        self.up2 = Up(in_channels=channels[1], out_channels=channels[0], kernel=kernel, dilation=dilation, groups=groups,
                      upsample_kernel = upsample_kernel, upsample_stride = upsample_stride, padding_down=1, padding_up=2)

        self.out_conv = nn.Conv3d(channels[0], out_channels, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.MaxPool3d(max_pool_kernel)

        self.fz = f(self.down2_fz, self.down3_fz, self.up1_fz, self.max_pool)
        self.fh = f(self.down2_fh, self.down3_fh, self.up1_fh, self.max_pool)

    def forward(self, image):

        outputs = []

        for t in range(10):
            if t == 0:
                s_t = torch.zeros([1, 5, image.shape[2], image.shape[3], image.shape[4]]).cuda()

            x = torch.cat((image, s_t), dim=1)

            x = self.down1(x)
            a = x.clone()
            x = self.max_pool(x)

            # Recurrent Bit!!!
            h = self.tanh(self.fh(x))
            if t == 0:
                h_t = torch.ones(h.shape).cuda()

            z = self.sigmoid(self.fz(x))
            h_t = (h_t * z) + (-1 * z * h)

            #
            x = self.up2(h_t, a)
            x = self.out_conv(x)
            s_t = x

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


class RDCNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RDCNet, self).__init__()

        complexity = 10

        self.strided_conv = nn.Conv3d(in_channels, complexity, kernel_size=3, stride=2, padding=1)
        self.RDCblock = RDCBlock(complexity)
        self.out_conv = nn.Conv3d(complexity, out_channels=complexity, kernel_size=3,padding=1)
        self.transposed_conv = nn.ConvTranspose3d(in_channels=complexity, out_channels=out_channels,
                                                  stride=(2,2,2), kernel_size=(4, 4, 4), padding=(1,1,1))

    def forward(self, x):
        x = self.strided_conv(x)
        for t in range(10):
            if t == 0:
                y = torch.zeros(x.shape).cuda()
            in_ = torch.cat((x, y), dim=1)
            y = self.RDCblock(in_) + y
        y = self.out_conv(y)
        return self.transposed_conv(y)




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
                 padding=None
                 ):
        super(Down, self).__init__()
        if padding is None:
            padding = 0

        self.conv1 = nn.Conv3d(in_channels,
                                       out_channels,
                                       kernel['conv1'],
                                       dilation=dilation['conv1'],
                                       groups=groups['conv1'],
                                       padding=padding)

        self.conv2 = nn.Conv3d(out_channels,
                                       out_channels,
                                       kernel['conv2'],
                                       dilation=dilation['conv2'],
                                       groups=groups['conv2'],
                                       padding=1)

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
                 padding_down=None,
                 padding_up=None
                 ):

        super(Up, self).__init__()

        if padding_down is None:
            padding_down=0
        if padding_up is None:
            padding_up=0

        self.conv1 = nn.Conv3d(in_channels,
                                   out_channels,
                                   kernel['conv1'],
                                   dilation=dilation['conv1'],
                                   groups=groups['conv1'],
                                   padding=padding_down)
        self.conv2 = nn.Conv3d(out_channels,
                                   out_channels,
                                   kernel['conv2'],
                                   dilation=dilation['conv2'],
                                   groups=groups['conv2'],
                                   padding=padding_down)

        self.up_conv = nn.ConvTranspose3d(in_channels,
                                         out_channels,
                                         upsample_kernel,
                                         stride=upsample_stride,
                                         padding=padding_up)
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


class StackedDilation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple,
                 ):

        super(StackedDilation, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=1, padding=2)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=2, padding=4)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=3, padding=6)
        self.conv4 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=4, padding=8)
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, dilation=5, padding=10)
        self.out_conv = nn.Conv3d(out_channels*5, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        out = torch.cat((x1, x2, x3, x4, x5),dim=1)
        out = self.out_conv(out)
        return out


class RDCBlock(nn.Module):
    def __init__(self, in_channels):

        super(RDCBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels*2, in_channels, kernel_size=1)
        self.grouped_conv = StackedDilation(in_channels, in_channels, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.grouped_conv(x)
        return x

