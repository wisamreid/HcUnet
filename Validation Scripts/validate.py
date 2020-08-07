import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import glob
import re

path = '/media/chris/Padlock_3/ToAnalyze/'

folders = glob.glob(path+'*_cellBycell/')

image_name = []
gfp_list = []
myo_list = []
dapi_list = []
actin_list = []
for f in folders:
    print(f)
# 8 and 10 for cmv8 m3
# 9, 11 for cmv8 m4

# folders=[folders[8],folders[10]]
keep_mask = False
for f in folders:

    # plt.figure()
    # plt.yscale('log')
    # plt.xlabel('GFP Intensity')
    # plt.ylabel('Num Pixels (Log)')

    name = os.path.basename(f[0:-1:1])
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
    all_cells = pickle.load(open(f+'all_cells.pkl', 'rb'))
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

    gfp = np.array(gfp).flatten()
    myo = np.array(myo).flatten()
    dapi = np.array(dapi).flatten()
    actin = np.array(actin).flatten()
    print('\b \b'*n,end='')

    gfp_list.append(gfp)
    myo_list.append(myo)
    dapi_list.append(dapi)
    actin_list.append(actin)

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
    print('Done')

plt.figure()
plt.boxplot(gfp_list)
ax = plt.gca()
plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=45)
plt.ylabel('GFP cell average intensity')
plt.title('GFP (cell by cell)')
plt.show()


plt.figure()
plt.boxplot(myo_list)
ax = plt.gca()
plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=45)
plt.ylabel('myo7a cell average intensity')
plt.title('myo7a (cell by cell)')
plt.show()

plt.figure()
plt.boxplot(dapi_list)
ax = plt.gca()
plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=45)
plt.ylabel('DAPI cell average intensity')
plt.title('DAPI (cell by cell)')
plt.show()

plt.figure()
plt.boxplot(actin_list)
ax = plt.gca()
plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=45)
plt.ylabel('Actin cell average intensity')
plt.title('Actin (cell by cell)')
plt.show()
