import numpy as np
import pickle
import os
import mask
import skimage.io as io
import glob


root_dir = '/Users/chrisbuswinka/Dropbox (Partners HealthCare)/HcUnet/Data/Originals/'
folders = glob.glob(root_dir+'*/')
for f in folders:
    filename = glob.glob(f+'*.labels.tif')[0]

    filename = os.path.splitext(filename)[0]
    filename = os.path.splitext(filename)[0]
    print(filename)
    makemask = mask.makeMask(erosion=False)
    makepwl = mask.makePWL()

    colormask = makemask(filename + '.labels.tif')
    io.imsave(filename+'.colormask.tif', colormask)

    if not os.path.exists(filename+'.pwl.pkl'):
        pwl = makepwl(filename+'.colormask.tif')
        pickle.dump(pwl, open(filename+'.pwl.pkl','wb'))

    cm = np.copy(colormask)
    bw_mask = mask.colormask_to_mask(cm)
    print(bw_mask.shape)

    io.imsave(filename+'.mask.tif', bw_mask)
