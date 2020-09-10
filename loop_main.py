import os
import glob
from hcat.main import analyze
import torch
import gc
path = '/media/chris/Padlock_3/ToAnalyzeGain/'
image_files = glob.glob(path + '*.tif')
os.chdir(path)

gfp = []

for image_loc in image_files:
    print(f'Analyzing: {image_loc}')
    foldername = os.path.splitext(image_loc)[0]
    try:
        os.mkdir(foldername+'_cellBycell')
        os.chdir(foldername+'_cellBycell')
        os.mkdir('./maskfiles')
    except:
        print(f'ERROR: {image_loc}, continuing...')
        continue
    try:
        analyze(image_loc, numchunks=6, save_plots=True, path_chunk_storage='./maskfiles/')
    except RuntimeError:
        analyze(image_loc, numchunks=5, save_plots=True, path_chunk_storage='./maskfiles/')
    os.chdir('..')
    print('')
    gc.collect(2)
    torch.cuda.empty_cache()


