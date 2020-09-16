import os
import glob
from hcat.main import analyze
import torch
import gc
path = '/media/chris/Padlock_3/ToAnalyze/'
image_files = glob.glob(path + '*.tif')
os.chdir(path)

gfp = []

for image_loc in image_files:
    print(f'Analyzing: {image_loc}')
    foldername = os.path.splitext(image_loc)[0]
    try:
        os.mkdir(foldername+'_cellBycell')
    except:
        print(f'ERROR: {image_loc}, continuing...')
        # continue

    os.chdir(foldername + '_cellBycell')
    os.mkdir('./maskfiles')
    try:
        analyze(image_loc, numchunks=6, save_plots=False, path_chunk_storage='./maskfiles/')
    except RuntimeError:
        analyze(image_loc, numchunks=7, save_plots=False, path_chunk_storage='./maskfiles/')



    os.chdir('..')
    print('')
    gc.collect(2)
    torch.cuda.empty_cache()


