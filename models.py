import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

class Unet3D(torch.nn.Module):
    def __init__(self,
                 n_classes=2,
                 kernel_size=3,
                 up_sample_kernel_size=2,
                 max_pool_kernel_size=2,
                 fs=[64, 128, 256, 512, 1024]):
        # super(Unet3D, self).__init__()

        self.down_step1_1 = torch.nn.Conv3d( 3, fs[0], kernel_size=kernel_size, padding=0)
        self.down_step1_2 = torch.nn.Conv3d(fs[0], fs[0], kernel_size=kernel_size, padding=0)

        self.down_step2_1 = torch.nn.Conv3d(fs[0], fs[1], kernel_size=kernel_size, padding=0)
        self.down_step2_2 = torch.nn.Conv3d(fs[1], fs[1], kernel_size=kernel_size, padding=0)

        self.down_step3_1 = torch.nn.Conv3d(fs[1], fs[2], kernel_size=kernel_size, padding=0)
        self.down_step3_2 = torch.nn.Conv3d(fs[2], fs[2], kernel_size=kernel_size, padding=0)

        self.down_step4_1 = torch.nn.Conv3d(fs[2],fs[3], kernel_size=kernel_size, padding=0)
        self.down_step4_2 = torch.nn.Conv3d(fs[3],fs[3], kernel_size=kernel_size, padding=0)

        self.bottom_step5_1 = torch.nn.Conv3d(fs[3], fs[4], kernel_size=kernel_size, padding=0)
        self.bottom_step5_2 = torch.nn.Conv3d(fs[4], fs[4], kernel_size=kernel_size, padding=0)

        self.up_conv_1 = torch.nn.ConvTranspose3d(fs[4], fs[3], kernel_size=up_sample_kernel_size, padding=0)
        self.up_step6_1 = torch.nn.ConvTranspose3d(fs[4],fs[3], kernel_size=kernel_size, padding=0)
        self.up_step6_2 = torch.nn.ConvTranspose3d(fs[3],fs[3], kernel_size=kernel_size, padding=0)

        self.up_conv_2 = torch.nn.ConvTranspose3d(fs[3], fs[2], kernel_size=up_sample_kernel_size, padding=0)
        self.up_step7_1 = torch.nn.Conv3d(fs[3], fs[2], kernel_size=kernel_size, padding=0)
        self.up_step7_2 = torch.nn.Conv3d(fs[2],fs[2], kernel_size=kernel_size, padding=0)

        self.up_conv_3 = torch.nn.ConvTranspose3d(fs[2], fs[1], kernel_size=up_sample_kernel_size, padding=0)
        self.up_step8_1 = torch.nn.Conv3d(fs[2], fs[1], kernel_size=kernel_size, padding=0)
        self.up_step8_2 = torch.nn.Conv3d(fs[1], fs[1], kernel_size=kernel_size, padding=0)

        self.up_conv_4 = torch.nn.ConvTranspose3d(fs[1], fs[0], kernel_size=up_sample_kernel_size, padding=0)
        self.up_step9_1 = torch.nn.Conv3d(fs[1], fs[0], kernel_size=kernel_size, padding=0)
        self.up_step9_2 = torch.nn.Conv3d(fs[0], fs[0], kernel_size=kernel_size, padding=0)

        self.mask_conv = torch.nn.Conv3d(fs[0], n_classes, kernel_size=1, padding=0)

        self.max_pool = torch.nn.MaxPool3d(max_pool_kernel_size)

    def forward(self, image):

        step1_image = F.relu(self.down_step1_1(image))
        step1_image = F.relu(self.down_step1_2(step1_image))

        print(f'Step 1 Shape: {step1_image.shape}')

        step2_image = self.max_pool(step1_image)
        step2_image = F.relu(self.down_step2_1(step2_image))
        step2_image = F.relu(self.down_step2_2(step2_image))

        print(f'Step 2 Shape: {step2_image.shape}')

        step3_image = self.max_pool(step2_image)
        step3_image = F.relu(self.down_step3_1(step3_image))
        step3_image = F.relu(self.down_step3_2(step3_image))

        print(f'Step 3 Shape: {step3_image.shape}')

        step4_image = self.max_pool(step3_image)
        step4_image = F.relu(self.down_step4_1(step4_image))
        step4_image = F.relu(self.down_step4_2(step4_image))

        print(f'Step 4 Shape: {step4_image.shape}')

        step5_image = self.max_pool(step4_image)
        step5_image = F.relu(self.bottom_step5_1(step5_image))
        step5_image = F.relu(self.bottom_step5_2(step5_image))

        print(f'Step 5 Shape: {step5_image.shape}')

        up_image = self.up_conv_1(step5_image)
        up_image = torch.cat((up_image, step4_image), dim=2)
        up_image = F.relu(self.up_step6_1(up_image))
        up_image = F.relu(self.up_step6_2(up_image))

        print(f'Step 6 Shape: {up_image.shape}')

        up_image = self.up_conv_2(up_image)
        up_image = torch.cat((up_image, step3_image), dim=2)
        up_image = self.up_step7_1(up_image)
        up_image = self.up_step7_2(up_image)

        print(f'Step 7 Shape: {up_image.shape}')

        up_image = self.up_conv_3(up_image)
        up_image = torch.cat((up_image, step2_image), dim=2)
        up_image = F.relu(self.up_step8_1(up_image))
        up_image = F.relu(self.up_step8_2(up_image))

        print(f'Step 8 Shape: {up_image.shape}')

        up_image = self.up_conv_4(up_image)
        up_image = torch.cat((up_image, step1_image), dim=2)
        up_image = F.relu(self.up_step9_1(up_image))
        up_image = F.relu(self.up_step9_2(up_image))

        print(f'Step 9 Shape: {up_image.shape}')

        up_image = self.mask_conv(up_image)

        print(f'Mask Shape: {up_image.shape}')

        return up_image






