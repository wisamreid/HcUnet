import torch

def pad_image_with_reflections(image, pad_size=(30,30,6)):
    """
    Pads image according to Unet spec
    expect [z,y,x,c]

    :param image:
    :param pad_size:
    :return:
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Expected image to be of type torch.tensor not {type(image)}')
    for pad in pad_size:
        if pad % 2 != 0:
            raise ValueError('Padding must be divisible by 2')

    image_size = image.shape  # should be some flavor of [Batchsize, C, X, Y, Z]

    out_size = [image_size[0],  # Batch Size
                image_size[1],  # Color Channels
                image_size[2] + pad_size[0],  # x
                image_size[3] + pad_size[1],  # y
                image_size[4] + pad_size[2],  # z
                ]

    left_pad = image[:, :, 0:pad_size[0], :, :].flip(2)
    right_pad = image[:, :, -pad_size[0]::, :, :].flip(2)

    out = torch.cat((left_pad, image, right_pad), dim=2)

    bottom_pad = out[:, :, :, 0:pad_size[1], :].flip(3)
    top_pad = out[:, :, :, -pad_size[1]::, :].flip(3)

    out = torch.cat((bottom_pad, out, top_pad), dim=3)

    return out


