import numpy as np
from PIL import Image, TiffImagePlugin

loc = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged.tif'

a = Image.open(loc)
b = np.array(a.getdata(), np.uint16)