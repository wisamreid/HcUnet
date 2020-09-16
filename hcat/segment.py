import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage
import skimage.feature
import cv2

import hcat
from hcat import utils
from hcat.haircell import HairCell



def predict_segmentation_mask(unet, image, device, use_probability_map=False, mask_cell_prob_threshold=0.5):
    """
    Uses pretrained unet model to predict semantic segmentation of hair cells.

    ALGORITHM:
    Remove inf and nan ->
    apply padding ->
    calculate indexes for unet based on image.shape ->
    apply Unet on slices of image based on indexes ->
    take only valid portion of the middle of each output mask ->
    construct full valid mask ->
    RETURN mask


    BLUR PROBABILITY MAP???

    :param unet: Trained Unet Model from hcat.unet
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :param device: 'cuda' or 'cpu'
    :param mask_cell_prob_threshold: float between 0 and 1 : min probability of cell used to generate the mask
    :param use_probability_map: bool, if True, does not apply sigmoid function to valid out, returns prob map instead
    :return mask:

    """
    # Check for total cuda memory to avoid overflow later
    # Define useful variables
    # Common GPU Memory Sizes (4, 6, 8, 11)
    _eval_im_size = {'4': [128, 128, 6],  # In GB
                     '6': [300, 300, 6],
                     '8': [300, 300, 10],
                     '11': [350, 350, 15]}
    if hcat.__CUDA_MEM__:
        PAD_SIZE = (128, 128, 10)
        EVAL_IMAGE_SIZE = _eval_im_size[str(int(np.floor(hcat.__CUDA_MEM__/1e9)))]
    else:
        PAD_SIZE = [128, 128, 10]
        EVAL_IMAGE_SIZE = [300, 300, 15]

    mask = torch.zeros((1, 1, image.shape[2], image.shape[3], image.shape[4]), dtype=torch.float)
    im_shape = image.shape

    if im_shape[4] < EVAL_IMAGE_SIZE[2]:
        EVAL_IMAGE_SIZE[2] = im_shape[4]

    # inf and nan screw up model evaluation. Happens occasionally. Remove them!
    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 1

    # Apply Padding
    image = utils.pad_image_with_reflections(torch.as_tensor(image), pad_size=PAD_SIZE)

    #  We now calculate the indicies for our image
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[2], image.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[3], image.shape[3])
    z_ind = utils.calculate_indexes(PAD_SIZE[2], EVAL_IMAGE_SIZE[2], im_shape[4], image.shape[4])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind) * len(z_ind))-1

    # Loop and apply unet
    # (x,y,z)_ind are lists of lists [[0,100], [100,200], ... ]
    with torch.no_grad():
        for z in z_ind:
            for x in x_ind:
                for y in y_ind:
                    print(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}', end='', flush=True)

                    # t.to_tensor reshapes image to [B C X Y Z]!!!
                    padded_image_slice = image[:, :, x[0]:x[1], y[0]:y[1]:, z[0]:z[1]].float().to(device)

                    # Occasionally everything is just -1 in the whole mat. Skip for speed
                    if (padded_image_slice.float() == -1).all():
                        iterations += 1
                        print('\b \b' * len(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}'), end='')
                        continue

                    # Evaluate Unet with no grad for speed

                    valid_out = unet(padded_image_slice)

                    # Unet cuts off bit of the image. It is unavoidable so we add padding to either side
                    # We need to remove padding from the output of unet to get a "Valid" segmentation
                    valid_out = valid_out[:,:,
                                         PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                                         PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                                         PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2]]

                    # Perform an in place sigmoid function to save memory.
                    # 1/ (1+exp(-x))
                    valid_out.mul_(-1)
                    valid_out.exp_()
                    valid_out.add_(1)
                    valid_out.pow_(-1)

                    # Take pixels that are greater than 75% likely to be a cell.
                    if not use_probability_map:
                        valid_out.gt_(mask_cell_prob_threshold)  # Greater Than
                        valid_out = valid_out.type(torch.uint8)
                        if mask.dtype != torch.uint8:
                            mask = mask.type(torch.uint8)

                    # Add the valid unet segmentation to the larger mask matrix
                    try:
                        mask[:, :, x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                                   y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                                   z[0]:z[0]+EVAL_IMAGE_SIZE[2]] = valid_out
                    except IndexError:
                        raise RuntimeError(f'Amount of padding is not sufficient.\nvalid_out.shape: {valid_out.shape}\neval_image_size: {EVAL_IMAGE_SIZE} ')
                    except RuntimeError:
                        raise RuntimeError(f'Amount of padding is not sufficient.\nvalid_out.shape: {valid_out.shape}\neval_image_size: {EVAL_IMAGE_SIZE} '
                                           f'\npadded_image_slice.shape{padded_image_slice.shape} ')

                    iterations += 1
                    print('\b \b'*len(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}'), end='')

    return mask


def predict_cell_candidates(image, model, candidate_list = None, initial_coords=(0,0)):
    """
    Takes in an image in torch spec from the dataloader for unet and performs the 2D search for hair cells on each
    z plane, removes duplicates and spits back a list of hair cell locations, row identities, and probablilities

    Steps:
    process image ->
    load model ->
    loop through each slice ->
    compile list of dicts ->
    for each dict add a cell candidate to a master list ->
    if a close cell candidate has a higher probability, replace old candidate with new one ->
    return lists of master cell candidtates


    :param image:
    :param model:
    :return:
    """
    # Define some useful stuff
    _eval_im_size = {'4': (128, 128),  # In GB
                     '6': (300, 300),
                     '8': (500, 500),
                     '11': (600, 600)}
    if hcat.__CUDA_MEM__:
        PAD_SIZE = (24, 24)
        EVAL_IMAGE_SIZE = _eval_im_size[str(int(np.floor(hcat.__CUDA_MEM__/1e9)))]
    else:
        PAD_SIZE = (24, 24)
        EVAL_IMAGE_SIZE = (500, 500)

    im_shape = image.shape[2::]

    # Calculate mini chunk indexes
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], image.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], image.shape[3])

    total_iter = len(x_ind) * len(y_ind) * image.shape[-1]
    iter = 0

    # Loop through everything!
    with torch.no_grad():
        for z in range(image.shape[-1]):  # ASSUME TORCH VALID IMAGES [B, C, X, Y, Z]
            for x in x_ind:
                for y in y_ind:

                    info_string = f'{" "*len(str(total_iter))}{iter}/{total_iter}'
                    print(info_string, end='', flush=True)

                    # Take a mini chunk out of the original image
                    image_z_slice = image[:, :, x[0]:x[1], y[0]:y[1]:, z]

                    # Apply faster rcnn prediction model to this
                    predicted_cell_locations = model(image_z_slice)

                    # We're only doing a batch size of 1, take the first dict of results
                    predicted_cell_locations = predicted_cell_locations[0]  # list of dicts (batch size) with each dict containing 'boxes' 'labels' 'scores'

                    # We need to know what z plane these cells were predicted, add this as a new index to the dict
                    z_level = torch.ones(len(predicted_cell_locations['scores'])) * z
                    predicted_cell_locations['z_level'] = z_level

                    # We need everything to be a float and on the cpu()
                    for name in predicted_cell_locations:
                        predicted_cell_locations[name] = predicted_cell_locations[name].cpu().float()

                    # Because we take mini chunks, we need to combine all the cells to a single list
                    # This is done by hcat.utils.merge_cell_candidates
                    if not candidate_list:
                        candidate_list = predicted_cell_locations
                    else:
                        # We need to know which mini chunk is is, pass initial_coords as a tuple to make the fn aware
                        candidate_list = utils.merge_cell_candidates(candidate_list,
                                                                     predicted_cell_locations,
                                                                     initial_coords=(x[0], y[0]))

                    iter += 1
                    print('\b \b'*len(info_string), end='')

    return candidate_list


def generate_unique_segmentation_mask_from_probability(predicted_semantic_mask: np.ndarray,
                                                       predicted_cell_candidate_list: list,
                                                       image: np.ndarray,
                                                       cell_prob_threshold=.95, mask_prob_threshold=0.5):
    """


    :param predicted_semantic_mask:
    :param predicted_cell_candidate_list:
    :param image:
    :param cell_prob_threshold:
    :param mask_prob_threshold:
    :return:
    """
    # Define some useful stuff
    # Get memory usage - May need to fine tune later
    if np.round(hcat.__CPU_MEM__/1e9) >= 16:
        PAD_SIZE = [56, 56]
        EVAL_IMAGE_SIZE = [1212, 1212]
    else:
        PAD_SIZE = [64, 64]
        EVAL_IMAGE_SIZE = [412, 412]

    # If the pad + eval image size is larger than the image, just set the pad to 1, and the eval to the im size
    for dim, ps in enumerate(EVAL_IMAGE_SIZE):
        dim += 2
        shape = image.shape
        if shape[dim] < ps + 2*PAD_SIZE[dim-2]:
            EVAL_IMAGE_SIZE[dim-2] = shape[dim]
            PAD_SIZE[dim-2] = 1


    iterations = 0
    unique_cell_id = 2 # 1 is reserved for background
    cells = []
    expand_z = hcat.__expand_z__ # dialate the z to try and account for nonisotropic zstacks
    if expand_z < 1:
        raise ValueError('Cant expand by less than 1.... you goon...')


    im_shape = predicted_semantic_mask.shape[2::]
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], predicted_semantic_mask.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], predicted_semantic_mask.shape[3])

    max_iter = (len(x_ind) * len(y_ind))-1
    unique_mask = np.zeros(predicted_semantic_mask.shape, dtype=np.int32)
    seed = np.zeros(unique_mask.shape, dtype=np.int)

    if predicted_cell_candidate_list['boxes'] is None:
        Warning(f'No Predicted cells: {len(predicted_cell_candidate_list)}')
        return unique_mask, seed

    if len(predicted_cell_candidate_list['scores']) == 0:
        return unique_mask, seed

    if predicted_semantic_mask.dtype == np.float32:
        USE_PROB_MAP = True
    elif predicted_semantic_mask.dtype == np.uint8:
        USE_PROB_MAP = False
    else:
        raise ValueError(f'Unknown predicted_semantic_mask dtype. {type(predicted_semantic_mask)}, '
                         f'{predicted_semantic_mask.dtype} ')

    z = predicted_cell_candidate_list['z_level'].cpu().numpy()
    prob = predicted_cell_candidate_list['scores'].cpu().numpy()
    boxes = predicted_cell_candidate_list['boxes'][prob > cell_prob_threshold]
    z = z[prob > cell_prob_threshold]
    prob = prob[prob > cell_prob_threshold]

    # Try to basically remove boxes that arent likely to be over a cell.
    x1,y1,x2,y2 = boxes.T
    centers = [np.round(x1+(x2-x1)/2).tolist(), np.round(y1+(y2-y1)/2).tolist(), predicted_cell_candidate_list['z_level'].tolist()]
    centers = np.array(centers)
    ind = np.zeros(len(x1))
    iter = 0
    for x0, y0, z0 in centers.T:
        print(x0,y0,z0)
        try:
            if predicted_semantic_mask[0,0,int(x0), int(y0), int(z0)] > 0.5:
                ind[iter] = 1
                iter += 1
        except IndexError:
            iter += 1

    z = z[ind > 0]
    prob = prob[ind > 0]

    unique_z, counts = np.unique(z, return_counts=True)
    best_z = 0
    best_z_avg = 0
    for uni in unique_z:
        if prob[z==uni].mean() > best_z_avg:
            best_z = uni
            best_z_avg = prob[z==uni].mean()

    # We seem to have issues with watershed when seed values are way different from each other???
    # Sort by x0 to try and reduce this problem...
    # Maybe didnt help all that much. Still pretty though, im going to leave it...
    sorted, indices = torch.sort(predicted_cell_candidate_list['boxes'],dim=0)
    for key in predicted_cell_candidate_list:
        if predicted_cell_candidate_list[key].ndim == 2:
            predicted_cell_candidate_list[key] = predicted_cell_candidate_list[key][indices[:,0], :]
        else:
            predicted_cell_candidate_list[key] = predicted_cell_candidate_list[key][indices[:, 0]]

    for i, (y1, x1, y2, x2) in enumerate(predicted_cell_candidate_list['boxes']):

        # There are various conditions where a box is invalid
        # in these cases, do not place a seed and skip the box
        if x1 > image.shape[2]:
            continue  # box is outside x dim
        elif y1 > image.shape[3]:
            continue  # box is outside y dim
        elif predicted_cell_candidate_list['scores'][i] < cell_prob_threshold:
            continue  # box probability is lower than predefined threshold
        elif predicted_cell_candidate_list['z_level'][i] < best_z-hcat.__z_tolerance__:
            continue  # box is on the wrong z plane within a tolerance
        elif predicted_cell_candidate_list['z_level'][i] > best_z+hcat.__z_tolerance__:
            continue  # box is on the wrong z plane within a tolerance

        # in the cases where a box is clipping the outside, crop it to the edges
        if x2 > image.shape[2]:
            x2 = torch.tensor(image.shape[2] - 1).float()
        elif y2 > image.shape[3]:
            y2 = torch.tensor(image.shape[3] - 1).float()

        # Each box is a little to conservative in its estimation of a hair cell
        # To compensate, we add dx and dy to the corners to increase the size 
        dx = [-10, 10]
        dy = [-10, 10]

        if (x1 + dx[0]) < 0:
            dx[0] = x1
        if (y1 + dy[0]) < 0:
            dy[0] = y1
        if (x2 + dx[1]) > image.shape[2]:
            dx[1] = image.shape[2] - x2
        if (y2 + dy[1]) > image.shape[3]:
            dy[1] = image.shape[3] - y2

        x1 = x1.clone()
        x2 = x2.clone()
        y1 = y1.clone()
        y2 = y2.clone()

        x1 += dx[0]
        x2 += dx[1]
        y1 += dy[0]
        y2 += dy[1]

        x1 = int(torch.round(x1.clone()))
        x2 = int(torch.round(x2.clone()))
        y1 = int(torch.round(y1.clone()))
        y2 = int(torch.round(y2.clone()))

        # EXPERIMENTAL - TRY PLACEING EVERY SEED ON THE SAME Z PLANE, SHOULD BE BETTER I THINK
        # Here we place a seed value for watershed at each point of the valid box
        seed_square = 6
        for i in range(seed_square):
            for r0 in range(seed_square):
                for r1 in range(seed_square):
                    seed[0, 0, int(np.round(x1+(x2-x1)/2))+r0, int(np.round(y1+(y2-y1)/2))+r1, int(best_z) + i] = int(unique_cell_id)
        unique_cell_id += 1
    
    # Now we can loop through mini chunks and apply watershed to the mask or probability map
    for x in x_ind:
        for y in y_ind:
            print(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}', end='', flush=True)
            
            # Take a slice for evaluation
            mask_slice = predicted_semantic_mask[:, :, x[0]:x[1], y[0]:y[1]:, :]
            
            # Preallocate some matrices
            distance = np.zeros(mask_slice.shape)
            seed_slice = np.zeros(mask_slice.shape).astype(np.int)
            distance_expanded = np.zeros((distance.shape[2], distance.shape[3], distance.shape[4] * expand_z))
            seed_slice_expanded = np.zeros((distance.shape[2], distance.shape[3], distance.shape[4] * expand_z))
            mask_expanded = np.zeros((distance.shape[2], distance.shape[3], distance.shape[4] * expand_z))
            labels = np.zeros(distance.shape)

            # Context aware probability map
            # If true, watershed is done on the probability map, distance is now probability
            if USE_PROB_MAP and mask_slice.max() > 1:
                # Experimetnal
                mask_slice += 1e-8
                mask_slice -= np.min(mask_slice)
                mask_slice /= np.max(mask_slice)

                mask_slice_binary = mask_slice > mask_prob_threshold
                distance[0, 0, :, :, :] = mask_slice[0, 0, :, :, :]
                
            # If false, run a distance transform on the binary mask and run watershed on that
            else:
                mask_slice_binary = np.copy(mask_slice)
                for z in range(distance.shape[-1]):
                    distance[0, 0, :, :, z] = cv2.distanceTransform(mask_slice[0, 0, :, :, z].astype(np.uint8),
                                                                    cv2.DIST_L2, 5)

            # We want to run watershed with some padding
            # This will help us merge segmentations together later
            seed_slice[:, :, PAD_SIZE[0]:PAD_SIZE[0]+EVAL_IMAGE_SIZE[0], PAD_SIZE[1]:PAD_SIZE[1]+EVAL_IMAGE_SIZE[1], :] \
                = seed[:, :, x[0]+PAD_SIZE[0]:x[1]-PAD_SIZE[0]+1, y[0]+PAD_SIZE[1]:y[1]-PAD_SIZE[1]+1, :]

            # Confocal voxels are nonisotropic and the watershed algorithm cannot correct for this
            # We manually correct for this by copying z slices and expanding the z dim
            for i in range(distance.shape[4]):
                for j in range(expand_z):
                    distance_expanded[:, :, (expand_z * i + j)] = distance[0, 0, :, :, i]
                    seed_slice_expanded[:, :, (expand_z * i + j)] = seed_slice[0, 0, :, :, i]
                    mask_expanded[:, :, (expand_z * i + j)] = mask_slice_binary[0, 0, :, :, i]


            # # EXPERIMENTAL
            # # maybe by adjusting the probability map to include steep cuttoffs in gradient, we can improve watershed
            distance_expanded[distance_expanded < .2] = 0

            # # EXPERIMENTAL
            for i in range(hcat.__expand_mask__):
                mask_expanded = skimage.morphology.binary_dilation(mask_expanded)

            seed_slice_expanded[distance_expanded < 0.15] = 1

            # distance_expanded[distance_expanded > .85] = 1
            # distance_expanded[distance_expanded < 0.1]=0

            # Run the watershed algorithm
            # Seems to help when you square the distance function... creates steeper gradients????
            # compactness  > 0.8 is too much
            # Best is .03
            labels_expanded = skimage.segmentation.watershed((distance_expanded) * -1, seed_slice_expanded,
                                                    mask=mask_expanded,
                                                    connectivity=hcat.__conectivity__,
                                                    watershed_line=True, compactness=hcat.__compactness__)

            #Hminima
            # A/B test between matlab watershed and skimage
            # train unet to predict distance map not the class label


            labels_expanded[labels_expanded == 1] = 0

            # Remove correction for nonisotropic voxels from the output
            for i in range(labels.shape[4]):
                labels[0, 0, :, :, i] = labels_expanded[:, :, expand_z * i]

            # Squeeze down to a 3d uint32 matrix
            labels = labels[0, 0, :, :, :]

            # If we set any cell thats touching the edge to be zero, we can probabily merge better
            left = labels[0,:,:]
            right = labels[-1,:,:]
            top = labels[:,0,:]
            bottom = labels[:,-1,:]

            edge_cells = []

            for edge in [left,right,top,bottom]:
                unqiue = np.unique(edge)
                for id in unqiue:
                    labels[labels == id] = 0

            # Combine mini chunk to larger chunk
            unique_mask[0, 0, x[0]:x[1], y[0]:y[1]:, :][labels > 0] = labels[labels > 0]

            # Increment iterations
            iterations += 1
            print('\b \b' * len(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}'), end='')

    return unique_mask, seed


def generate_cell_objects(image: torch.Tensor,  unique_mask, cell_candidates, x_ind_chunk, y_ind_chunk):
    """
    Quick and dirty


            DOESNT WORK BECAUSE YOURE INFERING THE CENTERS BASED ON THE OBJECT MASK
            PASS IT THE CELL LIST INSTEAD AND GET THE CENTER FROM THERE
            YOU'RE TOO STRESSED TO TRY THIS RIGHT NOW
            TAKE A BREAK
            FOR YOURSELF
            FROM PAST CHRIS...




    :param image: [B,C,X,Y,Z] torch tensor
    :param unique_mask: [X,Y,Z] numpy array
    :return:
    """
    if unique_mask.ndim != 3:
        unique_mask = unique_mask[0,0,:,:,:]

    cell_ids = np.unique(unique_mask)
    indicies = np.indices(unique_mask.shape) # Should be 3xMxN
    cell_list = []
    num_cells = len(cell_ids)
    print(' ',end='', flush=True)

    for i, id in enumerate(cell_ids):
        if id == 0:
            continue
        s = f'{i}/{num_cells}'
        print(s, end='', flush=True)

        mask = (unique_mask == id)
        x_ind = indicies[0, :, :, :][mask]
        y_ind = indicies[1, :, :, :][mask]
        z_ind = indicies[2, :, :, :][mask]
        x = (x_ind.min(), x_ind.max())
        y = (y_ind.min(), y_ind.max())
        z = (z_ind.min(), z_ind.max())

        image_coords = [x[0], y[0], z[0], x[1], y[1], z[1]]

        center = [(x[0] + (x[1]-x[0])/2) + x_ind_chunk, y[0] + ((y[1]-y[0])/2) + y_ind_chunk, (z[1]-z[0])/2]
        # center = [x[0], y[0], z[0]]
        image_slice = image[:, :, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        mask = mask[x[0]:x[1], y[0]:y[1], z[0]:z[1]]

        cell = HairCell(image_coords=image_coords, center=center, image=image_slice, mask=mask, id=id)
        cell_list.append(cell)
        print('\b \b'*(len(s)-1) , end='')
        print('\b', end='')

    return cell_list