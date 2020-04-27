import torch


def pad_image_with_reflections(image, pad_size=(30, 30, 6)):
    """
    Pads image according to Unet spec
    expect [B, C, X, Y, Z]

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

    bottom_pad = out[:, :, :, :, 0:pad_size[2]].flip(4)
    top_pad = out[:, :, :, :, -pad_size[2]::].flip(4)

    out = torch.cat((bottom_pad, out, top_pad), dim=4)



    return out


def predict_mask(model, image, device):
    """
    Takes in a model and an image and applies the model to all parts of the image.

    ALGORITHM:
    apply padding ->
    calculate indexes for unet based on image.shape ->
    apply Unet on slices of image based on indexes ->
    take only valid portion of the middle of each output mask ->
    construct full valid mask ->
    RETURN mask

    :param model: Trained Unet Model from unet.py
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :return mask:

    """
    EVAL_IMAGE_SIZE = (250, 250, 12)
    PAD_SIZE = (100, 100, 10)

    mask = torch.zeros(image.shape)

    # Apply Padding
    padded_image = pad_image_with_reflections(image, pad_size=PAD_SIZE)


    #  We now calculate the indicies for our image
    x_ind = calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], image.shape[2], padded_image.shape[2])
    y_ind = calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], image.shape[3], padded_image.shape[3])
    z_ind = calculate_indexes(PAD_SIZE[2], EVAL_IMAGE_SIZE[2], image.shape[4], padded_image.shape[4])

    # Loop and apply unet
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            for k, z in enumerate(z_ind):
                padded_image_slice = padded_image[:,:,x[0]:x[1], y[0]:y[1]:, z[0]:z[1]]
                print(x,y,z,padded_image_slice.shape)
                with torch.no_grad():
                    out = model(padded_image_slice.float().to(device))

                valid_out  = out[:,:,
                             PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                             PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                             PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2],
                                 ]

                mask[:, :, x[0]:x[0]+valid_out.shape[2],
                           y[0]:y[0]+valid_out.shape[3],
                           z[0]:z[0]+valid_out.shape[4]] = valid_out

    return mask


def calculate_indexes(pad_size, eval_image_size, image_shape, dim_shape):
    ind_list = torch.arange(pad_size, image_shape, eval_image_size)
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i-1])-pad_size
        z2 = int(z-1)+pad_size
        ind.append([z1, z2])
    if not ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append([dim_shape - (eval_image_size+pad_size*2), dim_shape])
    else:
        z1 = dim_shape-(eval_image_size + pad_size*2)
        z2 = dim_shape-1
        ind.append([z1, z2])
    return ind

