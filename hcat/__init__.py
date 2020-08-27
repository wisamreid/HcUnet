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
