import os
import glob
from functools import partial

from utils import dataloader

from models.fpn_model import FPNModel
from models.unet_model import UNetModel

from utils.apply_color_palette import CARLA_LABEL_COLORS

if __name__ == '__main__':
    ##########################################################
    ################## ADJUSTABLE PARAMS #####################
    ##########################################################
    NETWORK   = 'FPNModel'         # FPNModel, UNetModel
    BACKBONE  = 'MobileNetV2'      # MobileNetV2, ResNet50
    N_CLASS   = 13                 # number of segmentation classes
    
    WEIGHTS     = './log/ms_carla_fpn_mnet_step_up22/seg_pred_00500_epochs'
    IMAGE_DIR   = './val/images/'
    RES_DIR     = './val/preds_test/'
    APPLY_COLOR = True

    CROP_BBOX     = [0, 0, 448, 800]    # start_y, start_x, delta_y, delta_x (y -> height, x -> width)
    CROP_BBOX_PIL = [CROP_BBOX[1], CROP_BBOX[0], CROP_BBOX[3], CROP_BBOX[2]] # PIL uses a different ordering.

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d)" % (NETWORK, BACKBONE, N_CLASS))    

    model.load_weights(WEIGHTS)
    model.predict_folder(IMAGE_DIR,
                         RES_DIR,
                         color_palette=CARLA_LABEL_COLORS,
                         crop_bbox=CROP_BBOX_PIL, 
                         resize_shape=[416,224])
