from hcat.rcnn import rcnn
from hcat.unet import Unet_Constructor as unet
from hcat.segment import predict_cell_candidates, predict_segmentation_mask, generate_cell_objects, \
        generate_unique_segmentation_mask_from_probability
from hcat.main import analyze
import psutil
import torch

mem = psutil.virtual_memory()
__CPU_MEM__ = mem.total

if torch.cuda.is_available():
    torch.cuda.init()
    __CUDA_MEM__ = torch.cuda.get_device_properties(0).total_memory
else:
    __CUDA_MEM__ = False

# Watershed Constants
__conectivity__ = 15
__compactness__ = 0.001
__expand_mask__ = 15

# Seed placement Constants
__expand_z__ = 1
__z_tolerance__ = 3


# Hcat.generate_unique_segmentation_mask_from_probability
__mask_prob_threshold__ = 0.2
__cell_prob_threshold__ = 0.35

# unique_mask, seed = hcat.generate_unique_segmentation_mask_from_probability(predicted_semantic_mask.numpy(),
#                                                                             predicted_cell_candidate_list,
#                                                                             image_slice,
#                                                                             cell_prob_threshold=hcat.__cell_prob_threshold__,
#                                                                             mask_prob_threshold=hcat.__mask_prob_threshold__)




