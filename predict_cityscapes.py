import os
import glob
from functools import partial

from utils import dataloader

from models.fpn_model import FPNModel
from models.unet_model import UNetModel

from utils.apply_color_palette import CITY_LABEL_COLORS

if __name__ == '__main__':
    ##########################################################
    ################## ADJUSTABLE PARAMS #####################
    ##########################################################
    NETWORK   = 'FPNModel'                         # FPNModel, UNetModel
    BACKBONE  = 'ResNet50'                         # MobileNetv2, ResNet50    
    N_CLASS   = 20                                 # number of segmentation classes
    
    WEIGHTS     = './log/city_fpn_resnet/seg_pred_00000_epochs'
    IMAGE_DIR   = '/shared/govvijay/cityscapes/leftImg8bit/val/'
    RES_DIR     = './val/preds_city/'
    APPLY_COLOR = True


    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d)" % (NETWORK, BACKBONE, N_CLASS))    
    
    model.load_weights(WEIGHTS)
    model.predict_folder(IMAGE_DIR,
                         RES_DIR,
                         color_palette=CITY_LABEL_COLORS,
                         crop_bbox=None, 
                         resize_shape=None)





        

        