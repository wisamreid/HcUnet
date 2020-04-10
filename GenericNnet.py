import torch
import torch.nn as nn
import torch.nn.functional as F

class GenericNnet(nn.Module):
    def __init__(self, conv_functions=(nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d),
                 in_channels=3,
                 out_channels=2,
                 feature_sizes=[64, 128, 256, 512, 1024],
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
        super(GenericNnet, self).__init__()

        # Convert to dict of parameters
        #  In order to allow for multiple values passed to the first and second step of each convolution,
        #  we construct a tuple wit values of 'conv1' and 'conv2 denoting the parameter for each step

        if type(kernel) is tuple:
            kernel = {'conv1': kernel, 'conv2': kernel}
        if type(dilation) is int or type(dilation) is tuple:
            dilation = {'conv1': dilation, 'conv2': dilation}
        if type(groups) is int or type(groups) is tuple:
            groups = {'conv1': groups, 'conv2': groups}


        if len(feature_sizes) < 2:
            raise ValueError(f'The Number of Features must be at least 2, not {len(feature_sizes)}')

        for i, f in enumerate(feature_sizes[0:-1:1]):
            assert f*2 == feature_sizes[i+1], \
                f'Feature Sizes must be multiples of two from each other: {f} != {feature_sizes[i-1]}*2'

        self.first_down_conv = []
        # self.second_down_conv = []
        self.first_up_conv = []
        # self.second_up_conv = []
        self.upsample_conv = []

        # Assign Functions for first convolution
        self.first_up_conv.append(conv_functions[0](in_channels,
                                                      feature_sizes[0],
                                                      kernel['conv1'],
                                                      dilation=dilation['conv1'],
                                                      groups=groups['conv1'],
                                                      padding=0))

        # self.second_down_conv.append(conv_functions[0](feature_sizes[0],
        #                                                feature_sizes[0],
        #                                                kernel['conv2'],
        #                                                dilation=dilation['conv2'],
        #                                                groups=groups['conv2'],
        #                                                padding=0))

        # Down Steps
        for i in range(len(feature_sizes)-2, -1, -1):

            self.first_down_conv.append(conv_functions[0](feature_sizes[i+1],
                                                          feature_sizes[i],
                                                          kernel['conv1'],
                                                          dilation=dilation['conv1'],
                                                          groups=groups['conv1'],
                                                          padding=0))

            # self.second_down_conv.append(conv_functions[0](feature_sizes[i],
            #                                                feature_sizes[i],
            #                                                kernel['conv2'],
            #                                                dilation=dilation['conv2'],
            #                                                groups=groups['conv2'],
            #                                                padding=0))

        # Up Steps
        for i in range(1, len(feature_sizes), 1):
            self.first_up_conv.append(conv_functions[0](feature_sizes[i-1],
                                                        feature_sizes[i],
                                                        kernel['conv1'],
                                                        dilation=dilation['conv1'],
                                                        groups=groups['conv1'],
                                                        padding=0))

            # self.second_up_conv.append(conv_functions[0](feature_sizes[i],
            #                                              feature_sizes[i],
            #                                              kernel['conv2'],
            #                                              dilation=dilation['conv2'],
            #                                              groups=groups['conv2'],
            #                                              padding=0))

        # Up-Convs
        for i in range(0, len(feature_sizes), 1): # Range Ignores weird stuff.
            self.upsample_conv.append(conv_functions[1](feature_sizes[i],
                                                        feature_sizes[i],
                                                        upsample_kernel,
                                                        stride=upsample_stride,
                                                        padding=0))

        self.max_pool = conv_functions[2](max_pool_kernel)
        self.segmentation_conv = conv_functions[0](feature_sizes[0], out_channels, 1, padding=0)

        # Have to make it a module list for model.to('cuda:0') to work
        self.first_down_conv = nn.ModuleList(self.first_down_conv)
        # self.second_down_conv = nn.ModuleList(self.second_down_conv)
        self.first_up_conv = nn.ModuleList(self.first_up_conv)
        # self.second_up_conv = nn.ModuleList(self.second_up_conv)
        self.upsample_conv = nn.ModuleList(self.upsample_conv)
        print(self.first_up_conv)
        print(self.first_down_conv)
        print(self.upsample_conv)

    def forward(self, x):


        step_counter = 0
        print('UP')
        print(self.first_up_conv)
        print(self.upsample_conv)
        for conv1, up_conv in zip(self.first_up_conv, self.upsample_conv):
            x = F.relu(conv1(x))
            print(f'Step: {step_counter}-1: {x.shape}')
            print(f'Step: {step_counter}-upconv: {x.shape} -> {up_conv(x).shape}')
            x = up_conv(x)
            # print(f'Step: {step_counter}-cat: {x.shape} -> {previous_image.shape}')
            #print(x.shape,self.crop(previous_image, x).shape )


            print(f'Step: {step_counter}-1: {x.shape}')

            step_counter += 1
        print('DOWN')
        # Go down the U: Encoding
        for conv1 in self.first_down_conv[0:-1:1]:
            print(f'Step: {step_counter}-1: {x.shape}')
            x = F.relu(conv1(x))
            print(f'Step: {step_counter}-1: {x.shape}')
            x = self.max_pool(x)
            step_counter += 1

        # Bottom of the U.
        x = F.relu(self.first_down_conv[-1](x))
        # x = F.relu(self.second_down_conv[-1](x))

        # Go Up the U: Decoding

        x = self.segmentation_conv(x)

        return x

    @staticmethod
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

        assert shape_x[1] == shape_y[1], f'Inputs do not have same number of feature dimmensions: {shape_x} | {shape_y}'

        if len(shape_x) == 4:
            cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1]
        if len(shape_x) == 5:
            cropped_tensor = x[:, :, 0:shape_y[2]:1, 0:shape_y[3]:1, 0:shape_y[4]:1]

        return cropped_tensor
