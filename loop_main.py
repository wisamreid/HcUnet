import os
import glob
from hcat.main import analyze
from hcat.utils import cells_to_csv
import torch
import pickle
import gc
import time
# path = '/media/DataStorage/ToAnalyze/'
# analyzed_path = '/media/DataStorage/ToAnalyze/'
path = '/media/DataStorage/ToAnalyze/'
analyzed_path = '/media/DataStorage/ToAnalyze/'
image_files = glob.glob(path + '*.tif')
os.chdir(path)
gfp = []

for image_loc in image_files:
    start_time = time.asctime()
    print('\x1b[1;32m' + f'Analyzing:' + '\x1b[0m' + f'{image_loc}')
    print(f'\t Start: '+ '\x1b[1;34m' +f'{start_time}'+ '\x1b[0m')
    foldername = os.path.splitext(image_loc)[0]
    try:
        os.mkdir(foldername+'_cellBycell')
    except:
        print('\x1b[3;33m' + f'Analysis dir already exists...'+ '\x1b[0m')

    os.chdir(foldername + '_cellBycell')


    print('\x1b[1;33m' + f'Creating CSV:' + '\x1b[0m' + f'{image_loc}')
    all_cells = pickle.load(open('all_cells.pkl','rb'))
    cells_to_csv(all_cells, 'all_cells.csv')
    print(f'DONE')

    if os.path.exists('./analysis.lock'):
        print('\x1b[3;31m' + f'Anaysis was previously computed. Skipping this image...' + '\x1b[0m')
        os.chdir('..')
        print(' ')
        continue

    try:
        os.mkdir('./maskfiles')
    except FileExistsError:
        print('\x1b[3;33m' +f'maskfiles dir already exists...'+ '\x1b[0m')


    try:
        analyze(image_loc, numchunks=6, save_plots=False, path_chunk_storage='./maskfiles/')
    except RuntimeError:
        analyze(image_loc, numchunks=6, save_plots=False, path_chunk_storage='./maskfiles/')


    end_time = time.asctime()

    if not os.path.exists('./analysis.lock'):
        out_str = f'Start: {start_time} \nEnd: {end_time}'
        file = open('analysis.lock', 'w')
        file.write(out_str)
        file.close()


    os.chdir('..')
    print('')
    gc.collect(2)
    torch.cuda.empty_cache()


