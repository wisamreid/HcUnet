import skimage.io as io
import os
import ray
import glob
from hcat.train import train_utils

basedir = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/validate/*.labels.tif'

ray.init()

mm = train_utils.makeMask(erosion=True)
mpwl = train_utils.makePWL()

images = glob.glob(basedir)
print(images)
results = []
#
#    NOTES SO YOU ONLY HAVE TO DO THIS ONCE
#    Please save the amira files as rgb tif's or else it wont work.
#
@ray.remote
def make_mask(image_path):
    image = mm(image_path)
    basename = os.path.splitext(image_path)[0]
    basename = basename
    io.imsave(basename+'.mask.tif', image)
    pwl = mpwl(basename+'.mask.tif')
    print(f'PWL MAX ray: {pwl.max()}')
    io.imsave(basename+'.pwl.tif', pwl)
    image = train_utils.colormask_to_mask(image)
    io.imsave(basename+'.mask.tif', image)
    print(basename, ' DONE')
    return image

for i in images:
    print(i)
    results.append(make_mask.remote(i))

ray.get(results)

