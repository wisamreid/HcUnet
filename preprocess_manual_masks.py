import skimage.io as io
import skimage.morphology
import pickle
import skimage.morphology
import matplotlib.pyplot as plt
import os
import ray
import glob
import numpy as np
from hcat.train import train_utils

basedir = 'Data/train/*.labels.tif'

ray.init()

mm = train_utils.makeMask(erosion=True)
mpwl = train_utils.makePWL()
com = train_utils.CalculateCenterOfMass()
pix2center = train_utils.VectorToCenter()

images = glob.glob(basedir)

results = []



center_of_mass, colormask  = com(images[0])
mask = mm(images[0])
test = pix2center(center_of_mass, colormask, mask)

# raise NotImplementedError


#    NOTES SO YOU ONLY HAVE TO DO THIS ONCE
#    Please save the amira files as rgb tif's or else it wont work.
#
@ray.remote
def make_mask(image_path):

    image = mm(image_path)
    center_of_mass, colormask = com(image_path)
    vector = pix2center(center_of_mass, colormask, image)

    basename = os.path.splitext(image_path)[0]
    basename = basename
    pickle.dump(vector, open(basename + '.vector.pkl', 'wb'))


    for i in range(5):
        center_of_mass = skimage.morphology.binary_erosion(center_of_mass > 0)

    io.imsave(basename+'.com.tif', (center_of_mass > 0).astype(np.uint8))

    # print(len(image.unique()), center_of_mass.max(), colormask.max(), vector.max(), )


    # io.imsave(basename+'.mask.tif', image)

    # pwl = mpwl(basename+'.mask.tif')
    # print(f'PWL MAX ray: {pwl.max()}')
    #
    # io.imsave(basename+'.pwl.tif', pwl)
    # image = train_utils.colormask_to_mask(image)
    # io.imsave(basename+'.mask.tif', image)
    # print(basename, ' DONE')

    return image

for i in images:
    print(i)
    results.append(make_mask.remote(i))

ray.get(results)

