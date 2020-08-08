import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.

from pspnet_model import PSPNetModel
from unet_model import UNetModel
from fpn_model  import FPNModel

for class_name in ['PSPNetModel', 'UNetModel', 'FPNModel']:
	print('Constructing ', class_name)
	model = eval('%s()' % class_name)
