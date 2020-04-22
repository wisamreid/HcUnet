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


def predict_mask(model, image):
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
    EVAL_IMAGE_SIZE = (200, 200, 12)
    PAD_SIZE = (100, 100, 6)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    model.to(device)
    model.eval()

    mask = torch.zeros(image.shape).to(device)

    # Apply Padding
    padded_image = pad_image_with_reflections(image, pad_size=PAD_SIZE)

    print(padded_image.shape)

    #  We now calculate the indicies for our image
    # X
    xlist = torch.arange(PAD_SIZE[0], image.shape[2], EVAL_IMAGE_SIZE[0])
    x_ind = []
    for i, x in enumerate(xlist):
        if i == 0:
            continue
        x1 = int(xlist[i-1])-PAD_SIZE[0]
        x2 = int(x-1)+PAD_SIZE[0]
        x_ind.append([x1, x2])
    x1 = padded_image.shape[2]-(EVAL_IMAGE_SIZE[0] + PAD_SIZE[0]*2)
    x2 = padded_image.shape[2]-1
    x_ind.append([x1, x2])
    print(x_ind)

    # Y
    ylist = torch.arange(PAD_SIZE[1], image.shape[3], EVAL_IMAGE_SIZE[1])
    y_ind = []
    for i, y in enumerate(ylist):
        if i == 0:
            continue
        y1 = int(ylist[i-1])-PAD_SIZE[1]
        y2 = int(y-1)+PAD_SIZE[1]
        y_ind.append([y1, y2])
    y1 = padded_image.shape[3]-(EVAL_IMAGE_SIZE[1] + PAD_SIZE[1]*2)
    y2 = padded_image.shape[3]-1
    y_ind.append([y1, y2])
    print(y_ind)

    # Z
    zlist = torch.arange(PAD_SIZE[2], image.shape[4], EVAL_IMAGE_SIZE[2])
    z_ind = []
    for i, z in enumerate(zlist):
        if i == 0:
            continue
        z1 = int(zlist[i-1])-PAD_SIZE[2]
        z2 = int(z-1)+PAD_SIZE[2]
        z_ind.append([z1, z2])
    if not z_ind:  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = EVAL_IMAGE_SIZE[2] + PAD_SIZE[2]
        z_ind.append([z1, z2])
        z_ind.append([padded_image.shape[4] - EVAL_IMAGE_SIZE[2], padded_image.shape[4]])
    else:
        z1 = padded_image.shape[4]-(EVAL_IMAGE_SIZE[2] + PAD_SIZE[2]*2)
        z2 = padded_image.shape[4]-1
        z_ind.append([z1, z2])
    print(z_ind)

    # Loop and apply unet
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            for k, z in enumerate(z_ind):
                padded_image_slice = padded_image[:,:,x[0]:x[1], y[0]:y[1]:, z[0]:z[1]]
                # out = model(padded_image_slice.float().to(device))
                out = padded_image_slice[:, 1, 0:230, 0:230, 0:14]
                valid_out  = out[:,:,
                             PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                             PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                             PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2],
                                 ]

                mask[:, :, x[0]:x[0]+EVAL_IMAGE_SIZE[0]-1,
                           y[0]:y[0]+EVAL_IMAGE_SIZE[1]-1,
                           z[0]:z[0]+EVAL_IMAGE_SIZE[2]-1] = valid_out



    return None