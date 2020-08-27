import os
import glob
from hcat.main import analyze

path = '/media/chris/Padlock_3/ToAnalyze/'
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
    analyze(image_loc)
    os.chdir('..')
    print('')


