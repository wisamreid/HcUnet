import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import re
from hcat import haircell
import io
import pickle
from sklearn.linear_model import LinearRegression
import pymc3 as pm
import pandas as pd
import plotarchive as pa


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

path_old = '/media/DataStorage/ToAnalyze/'
# path = '/media/DataStorage/ToAnalyzeLaserGain/'
path = '/media/DataStorage/AAV injection results/**/**/'



def sort_fun(input):
    if isinstance(input, tuple):
        input = input[0]
    gain = re.search('G\d\d\d?',input)[0]
    gain = gain[1::]

    promoter = re.search('(Full Len)|(Control)|(CMV\d\d?)',input)[0]
    if promoter == 'Full Len':
        promoter = 'CMVb'
    elif promoter == 'Control':
        promoter = 'CMVa'
    elif promoter == 'CMV11':
        promoter = 'CMV8'
    elif promoter == 'CMV8':
        promoter = 'CMV5'
    promoter = promoter[3::]

    animal = re.search('m\d',input)[0]
    animal = animal[1::]
    laser = re.search('L\d.\d\d?', input)[0]

    if laser[-1] != '0':
        laser = laser + '0'
    laser = laser[3::]

    date = re.search('P\d',input)[0]
    date = date[-1]

    return promoter + animal + gain + laser + date

folders = sorted(glob.glob(path + '*_cellBycell/'))
folders = folders + sorted(glob.glob(path_old + '*_cellBycell/'))
folders = sorted(folders)


promoter_list = []
id = []

for i, f in enumerate(folders):
    name = os.path.basename(f[0:-1:1])
    promoter = re.search('(Control)|(CMV\d?\d?)', name)[0]
    animal = re.search('m\d',name)[0]

    laser = re.search('(L\d.\d\d?)|(Laser \d.\d\d?)', name)
    if laser is not None:
        laser = laser[0]
        laser = re.search('\d.\d\d?', laser)
        laser = 'L' + laser[0]
    else:
        laser = 'L0.20'


    disection_day = re.search('P\d',name)
    if disection_day is not None:
        disection_day = disection_day[0]
    else:
        disection_day = 'P5'

    gain = re.search('(Gain\d?\d\d)|(G\d?\d\d)', name)
    if gain is not None:
        gain = gain[0]
        gain = 'G' + re.search('\d\d\d?', gain)[0]
    else:
        gain = 'G50'

    if re.search('Eric', name) is not None:
        promoter = 'Full Len'




    id.append(promoter + ' ' + disection_day + ' ' + animal + ' ' + gain + ' ' + laser )
    promoter_list.append(promoter)

folders = [x for _, x in sorted(zip(id, folders), key=sort_fun)]

promoter_list = np.unique(promoter_list)
# 8 and 10 for cmv8 m3
# 9, 11 for cmv8 m4

# folders=[folders[8],folders[10]]
traces = []
big_name_list = []
big_gfp_list = []

# for p in promoter_list:
keep_mask = False
image_name = []
gfp_list = []
myo_list = []
dapi_list = []
actin_list = []

name_list = []
gfp_dict = {}

################################################33
for i, f in enumerate(folders):

    name = os.path.basename(f[0:-1:1])
    promoter = re.search('(Control)|(CMV\d?\d?)', name)[0]

    animal = re.search('m\d', name)[0]
    laser = re.search('(L\d.\d\d?)|(Laser \d.\d\d?)', name)
    if laser is not None:
        laser = laser[0]
        laser = re.search('\d.\d\d?', laser)
        laser = 'L' + laser[0]
    else:
        laser = 'L0.25'

    disection_day = re.search('P\d', name)
    if disection_day is not None:
        disection_day = disection_day[0]
    else:
        disection_day = 'P5'

    gain = re.search('(Gain\d?\d\d)|(G\d?\d\d)', name)
    print(name, gain)
    if gain is not None:
        gain = gain[0]
        gain = 'G' + re.search('\d\d\d?', gain)[0]
    else:
        gain = 'G50'
    #
    if re.search('Eric', name) is not None:
        promoter = 'Full Len'
    #
    # if promoter != 'CMV11' and promoter != 'Control':
    #     continue
    # # if gain != 'G50':
    # #     continue
    # # # if disection_day == 'P7':
    # # #     continue
    # # if laser != 'L0.25':
    # #     continue



    id = promoter + ' ' + animal + ' ' + gain + ' ' + laser + ' ' + disection_day
    print(id)
    # print(f'{id}', end=' | ')
    try:
        all_cells = renamed_load(open(f+'all_cells.pkl', 'rb'))
    except (FileNotFoundError, EOFError):
        continue
    if len(all_cells) == 0:
        continue
    # if re.search('(Control)|(CMV\d?\d?)', name)[0] == p:
    #     continue
    # if re.search('Eric', name) is None:
    #     continue

    image_name.append(id)
    big_name_list.append(id)

    print('Loading cells...',end='')
    n = len('Loading cells...')


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

    gfp = np.array(np.array(gfp).flatten().__mul__(2**16) + 1).tolist()
    myo = np.array(myo).flatten()
    dapi = np.array(dapi).flatten()
    actin = np.array(actin).flatten()
    print('\b \b'*n,end='')

    gfp_list.append(gfp)
    big_gfp_list.append(gfp)
    name_list.append([promoter for i in range(len(gfp))])
    gfp_dict[id] = gfp
    myo_list.append(myo)
    dapi_list.append(dapi)
    actin_list.append(actin)
###################################################################

gfp_flat = [item for sublist in gfp_list for item in sublist]
name_flat = [item for sublist in name_list for item in sublist]

df = pd.DataFrame(data={'gfp':gfp_flat, 'id':name_flat})
keys = [key for key in gfp_dict]

# for p in df.id.unique():
#     data = df[df.id == p]
#     idx = np.zeros(len(data['id'].values), dtype=np.int)
#     for i, k in enumerate(keys):
#         idx[data['id'].values == k] = i
#
#         # with pm.Model() as gfp_model:
#         #     # Hyperparams
#         #     alpha = pm.Bound(pm.Normal, lower=0)('alpha', mu=1000, sd=10000)
#         #     beta = pm.HalfCauchy('beta', beta=5000)
#         #
#         #     # Model
#         #     mu = pm.Normal('mu_animal', mu=alpha, sd=beta, shape=len(image_name))
#         #     sigma = pm.HalfCauchy('sigma', beta=100, shape=len(image_name))
#         #
#         #     obs = pm.Normal('obs', mu=mu[idx], sd=sigma[idx], observed=data['gfp'].values)
#
#     with pm.Model() as gfp_model:
#         # Hyperparams
#         alpha = pm.Normal('alpha', mu=0, sd=2)
#         beta = pm.HalfCauchy('beta', beta=5)
#
#         # Model
#         mu = pm.Normal('mu_animal', mu=alpha, sd=beta, shape=len(image_name))
#         sigma = pm.HalfCauchy('sigma', beta=1, shape=len(image_name))
#
#         obs = pm.Normal('obs', mu=mu[idx], sd=sigma[idx], observed=data['gfp'].values)
#
#     with gfp_model:
#         trace = pm.sample(500, progressbar=True, chains=12, target_accept=.99)
#         print(pm.rhat(trace))
#         print(pm.summary(trace))
#         print(pm.bfmi(trace))
#         pm.traceplot(trace)
#         plt.tight_layout()
#         plt.show()
#         pm.plot_posterior(trace)
#         plt.tight_layout()
#         plt.show()
#
#     traces.append(trace['alpha'])

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
    # @pa.archive(filename='gfp_boxplots.pa')
    # def plot(x):
    #     plt.figure()
    #     plt.boxplot(x)
    #     ax = plt.gca()
    #     plt.xticks(np.arange(1, len(x)+1, 1), image_name, rotation=90)
    #     plt.ylabel('GFP cell average intensity')
    #     plt.title('GFP (cell by cell)')
    #     plt.show()
    # plot(gfp_list)

    # @pa.archive(filename='myo_boxplots.pa')
    # def plot(x):
    #     plt.figure()
    #     plt.boxplot(x)
    #     ax = plt.gca()
    #     plt.xticks(np.arange(1, len(x)+1, 1), image_name, rotation=90)
    #     plt.ylabel('GFP cell average intensity')
    #     plt.title('GFP (cell by cell)')
    #     plt.show()
    # plot(myo_list)

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

# tr = lambda x:  np.log10(x)
# for i, trace in enumerate(traces):
#     traces[i] = tr(trace)
#
# pm.plot_density(traces, data_labels = df.id.unique())
# plt.title('Estimated mean cell intensity of each CMV')
# plt.xlabel('Mean pixel intensity log10 (16bit)')
# plt.show()

plt.boxplot(big_gfp_list)
plt.xticks(np.arange(1, len(big_gfp_list)+1, 1), big_name_list, rotation=90)
plt.ylabel('Mean GFP Cell Intensity')
# plt.yscale('log')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('All_promoter_gfp.pdf')
plt.show()

gain = np.array([])
gfp_all = []
for l, name in zip(big_gfp_list, big_name_list):
    g = re.search('G\d\d?\d?',name)[0]
    g = int(g[1::])
    gain = np.concatenate((gain, np.ones(len(l)) * g), axis=0)
    gfp_all = gfp_all + l
gfp_all = np.array(gfp_all)

reg = LinearRegression().fit(gain.reshape(-1, 1) , gfp_all.reshape(-1, 1))
print(reg.coef_)
print(reg.intercept_)

plt.plot(gain, gfp_all,'.')
x = [0, 400]
plt.plot(x, [reg.intercept_[0] + x[0]*reg.coef_[0], reg.intercept_[0] + x[1]*reg.coef_[0]])
plt.xlabel('Gain')
plt.ylabel('GFP Cell Intensity')
plt.show()
# gain = np.array([])
# gfp_all = []
#
# for l, name in zip(big_gfp_list, big_name_list):
#     g = re.search('P\d\d?\d?',name)[0]
#     g = int(g[1::])
#     gain = np.concatenate((gain, np.ones(len(l)) * g), axis=0)
#     gfp_all = gfp_all + l
# gfp_all = np.array(gfp_all)
#
# reg = LinearRegression().fit(gain.reshape(-1, 1) , gfp_all.reshape(-1, 1))
# print(reg.coef_)
# print(reg.intercept_)
#
# plt.plot(gain, gfp_all,'.')
# x = [3, 9]
# plt.plot(x, [reg.intercept_[0] + x[0]*reg.coef_[0], reg.intercept_[0] + x[1]*reg.coef_[0]])
# plt.xlabel('Dissection Day')
# plt.ylabel('GFP Cell Intensity')
# plt.show()