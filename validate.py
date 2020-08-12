import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import re
from hcat import haircell
import io
import pickle
import pymc3 as pm
import pandas as pd


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "haircell":
            renamed_module = "hcat.haircell"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

path = '/media/chris/Padlock_3/ToAnalyze/'

folders = sorted(glob.glob(path+'*_cellBycell/'))


image_name = []
gfp_list = []
myo_list = []
dapi_list = []
actin_list = []
name_list = []

# 8 and 10 for cmv8 m3
# 9, 11 for cmv8 m4

# folders=[folders[8],folders[10]]
keep_mask = False

gfp_dict = {}

for i, f in enumerate(folders):
    if i == len(folders) - 1:
        continue

    name = os.path.basename(f[0:-1:1])

    if re.search('(Control)|(CMV\d?\d?)', name)[0] != 'CMV4':
        continue

    promoter = re.search('(Control)|(CMV\d?\d?)', name)[0]
    animal = re.search('m\d',name)[0]
    gain = re.search('Gain\d?\d?\d?',name)



    if re.search('Eric', name) is not None:
        promoter = promoter + ' Full'

    if gain is not None:
        gain = gain[0]
    else:
        gain = ''
    id = promoter + ' ' + animal + ' ' + gain
    print(f'{id}', end=' | ')
    image_name.append(id)

    print('Loading cells...',end='')
    n = len('Loading cells...')
    all_cells = renamed_load(open(f+'all_cells.pkl', 'rb'))
    print('\b \b'*n,end='')

    print('Compiling cell values...',end='')
    n = len('Compiling cell values...')
    gfp = []
    myo = []
    dapi = []
    actin = []
    for cell in all_cells:
        if not np.isnan(cell.gfp_stats['mean']):
            gfp.append(cell.gfp_stats['mean'])
            myo.append(cell.signal_stats['myo7a']['mean'])
            dapi.append(cell.signal_stats['dapi']['mean'])
            actin.append(cell.signal_stats['actin']['mean'])

    gfp = np.array(gfp).flatten().tolist()
    myo = np.array(myo).flatten()
    dapi = np.array(dapi).flatten()
    actin = np.array(actin).flatten()
    print('\b \b'*n,end='')

    gfp_list.append(gfp)
    name_list.append([id for i in range(len(gfp))])
    gfp_dict[id] = gfp
    myo_list.append(myo)
    dapi_list.append(dapi)
    actin_list.append(actin)

gfp_flat = [item for sublist in gfp_list for item in sublist]
name_flat = [item for sublist in name_list for item in sublist]

data = pd.DataFrame(data={'gfp':gfp_flat, 'id':name_flat})
keys = [key for key in gfp_dict]
idx = (data['id'].values == keys[0]).astype(np.uint8)
print(idx, idx.shape)

with pm.Model() as gfp_model:
    # Hyperparams
    alpha = pm.Normal('alpha', mu=0, sd=100)
    beta = pm.HalfCauchy('beta', beta=100)

    # tau = pm.Normal('tau', mu=0, sd=100)

    # Model
    mu = pm.Normal('mu_animal', mu=alpha, sd=beta, shape=2)
    sigma = pm.HalfCauchy('sigma', beta=100, shape=2)

    obs = pm.Normal('obs', mu=mu[idx], sd=sigma[idx], observed=data['gfp'].values * 2**16)

with gfp_model:
    trace = pm.sample(10000, progressbar=True, target_accept=.95)
    pm.traceplot(trace)
    plt.tight_layout()
    plt.savefig(f'{promoter}_traceplot.tif', dpi=300)
    plt.show()
    pm.plot_posterior(trace)
    plt.tight_layout()
    plt.savefig(f'{promoter}_posterior.tif', dpi=300)
    plt.show()
#
#
# coords = {"Level": ["Basement", "Floor"], "obs_id": np.arange(floor.size)}
# with pm.Model(coords=coords) as pooled_model:
#     floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
#     a = pm.Normal("a", 0.0, sigma=10.0, dims="Level")
#
#     theta = a[floor_idx]
#     sigma = pm.Exponential("sigma", 1.0)
#
#     y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
#
# pm.model_to_graphviz(pooled_model)

    # print(f'Loading Image...', end='')
    # n =len('Loading Image...')
    # image = io.imread(f[0:-1:1] + '.tif')[:,:,:,1] / 2**16
    # print('\b \b'*n, end='')
    #
    # print('Loading Mask...', end='')
    # n =len('Loading Mask...')
    # mask = io.imread(os.path.dirname(f) + '/test_mask.tif')
    # mask = mask > 0
    # if not keep_mask:
    #     keep_mask = np.copy(mask)
    # print('\b \b'*n, end='')

    # print('Indexing...',end='')
    # n =len('Indexing...')
    # gfp = image[keep_mask]
    # gfp_list.append(gfp)
    # pickle.dump(gfp, open(f'{id}.pkl', 'wb'))
    # plt.hist(gfp, bins=200, alpha=.5, range=[0,1])
    # del gfp
    # print('\b \b'*n, end='')

    # print('Smoothing Image...', end='')
    # smooth_image = scipy.ndimage.gaussian_filter(image, 3)
    # gfp_smooth = image[mask]
    # plt.hist(gfp_smooth, bins=200, alpha=.5, range=[0,1])
    # del gfp_smooth
    # n =len('Smoothing Image...')
    # print('\b'*n, end='')
    # print(' '*n, end='')
    # print('\b'*n, end='')
    # smooth_image=0

    # print('Dilating Mask...',end='')
    # gfp_dilated = image[scipy.ndimage.binary_dilation(mask, iterations=5)]
    # plt.hist(gfp_dilated, bins=200, alpha=.5, range=[0,1])
    # del gfp_dilated
    # n =len('Dilating Mask...')
    # print('\b'*n, end='')
    # print(' '*n, end='')
    # print('\b'*n, end='')
    #
    # print('Eroding Mask...',end='')
    # gfp_eroded = image[scipy.ndimage.binary_erosion(mask, iterations=5)]
    # plt.hist(gfp_eroded, bins=200, alpha=.5, range=[0,1])
    # del gfp_eroded
    # n =len('Eroding Mask...')
    # print('\b'*n, end='')
    # print(' '*n, end='')
    # print('\b'*n, end='')

    # plt.title(id)
    # plt.legend(['GFP','GFP Dilated', 'GFP Eroded'])
    # plt.savefig(f'{id}.png')
    # plt.show()
    #
    # del image
    # del mask
    # # del gfp
    # print('Done')
    # print('Done')
#
# plt.figure()
# plt.boxplot(gfp_list)
# ax = plt.gca()
# plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=90)
# plt.ylabel('GFP cell average intensity')
# plt.title('GFP (cell by cell)')
# plt.show()
#
#
# plt.figure()
# plt.boxplot(myo_list)
# ax = plt.gca()
# plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=90)
# plt.ylabel('myo7a cell average intensity')
# plt.title('myo7a (cell by cell)')
# plt.show()
#
# plt.figure()
# plt.boxplot(dapi_list)
# ax = plt.gca()
# plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=90)
# plt.ylabel('DAPI cell average intensity')
# plt.title('DAPI (cell by cell)')
# plt.show()
#
# plt.figure()
# plt.boxplot(actin_list)
# ax = plt.gca()
# plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=90)
# plt.ylabel('Actin cell average intensity')
# plt.title('Actin (cell by cell)')
# plt.show()
