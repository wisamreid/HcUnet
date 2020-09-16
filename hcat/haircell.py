import torch
import numpy as np


class HairCell:
    def __init__(self, image_coords, center, image, mask, id, type=None):
        self.image_coords = image_coords # [x1, y1, z1, x2, y2, z2]
        self.center = center  # [x,y,z]
        # self.mask = mask # numpy array | mask [x,y,z]
        # self.frequency = []
        self.type = type
        self.distance_from_apex = []
        self.unique_id = id
        self.is_bad = False
        self.signal_stats = {}
        self.volume = 0

        for z in range(mask.shape[-1]):
            #Assume 488 volume
            self.volume += 608e-9 * (mask[:,:,z] > 0).sum() * ((163e-9)**2)

        if isinstance(image, torch.Tensor):
            image = image.numpy()

        for i, channel in enumerate(['dapi', 'gfp', 'myo7a', 'actin']):
            if mask.sum() > 1:
                self.signal_stats[channel] = self._calculate_gfp_statistics(image, mask, i)  # green channel flattened array
            else:
                self.unique_mask = torch.zeros(10)
                self.is_bad = True
                self.signal_stats[channel] = {'mean': np.NaN, 'std': np.NaN, 'median': np.NaN}

        if mask.sum() > 1:
            self.gfp_stats = self._calculate_gfp_statistics(image, mask) # green channel flattened array
        else:
            self.unique_mask = torch.zeros(10)
            self.is_bad = True
            self.gfp_stats  = {'mean': np.NaN, 'std': np.NaN, 'median': np.NaN}

    def set_frequency(self, cochlea_curve, percentage):
        """
        Somehow take location of cell, and the spline fit to estimate cochelar frequency.

        :param location:
        :param cochlea_curveature:
        :return:
        """
        x = cochlea_curve[0,:]
        y = cochlea_curve[1,:]

        dist = np.sqrt((self.center[1] - x) ** 2 + (self.center[0] - y) ** 2)
        i = np.argmin(dist)

        self._place_percentage = percentage[i]
        self._closest_place = cochlea_curve[:, i]
        self.frequency = [self._closest_place, self._place_percentage]

    @staticmethod
    def _calculate_gfp_statistics(image, mask, channel=1):
        """
        calculates mean, std, and median gfp intensity for the cell and stores the values in a dict.

        :param image: numpy image
        :param mask:  numpy array of same size of image, type bool, used as index's for the image
        :return: dict{'mean', 'median', 'std'}
        """

        # channel = 1
        # 0:DAPI
        # 1:GFP
        # 2:MYO7a
        # 3:Actin

        mask = torch.tensor(mask > 0)

        if image.min() < 0:
            gfp = (image[0, channel, :, :, :][mask] * 0.5) + 0.5
        else:
            gfp = image[0, channel, :, :, :][mask]

        return {'mean': gfp.mean(), 'std': gfp.std(), 'median': np.median(gfp), 'num_samples': gfp.shape}

















