import os
import glob
from functools import partial

from utils import dataloader

from models.fpn_model import FPNModel
from models.unet_model import UNetModel

if __name__ == '__main__':
    ##########################################################
    ################## ADJUSTABLE PARAMS #####################
    ##########################################################
    NETWORK   = 'FPNModel'         # FPNModel, UNetModel
    BACKBONE  = 'MobileNetV2'      # MobileNetv2, ResNet50
    N_CLASS   = 13                 # number of segmentation classes
    
    WEIGHTS     = './log/folder/model_name'
    IMAGE_DIR   = './val/images/'
    RES_DIR     = './val/preds_test/'
    APPLY_COLOR = True

    CROP_BBOX     = [0, 0, 448, 800]    # start_y, start_x, delta_y, delta_x (y -> height, x -> width)
    CROP_BBOX_PIL = [CROP_BBOX[1], CROP_BBOX[0], CROP_BBOX[3], CROP_BBOX[2]] # PIL uses a different ordering.

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d)" % (NETWORK, BACKBONE, N_CLASS))
    parse_fun = partial(dataloader.parse_image, num_seg_classes=N_CLASS, crop_bbox=CROP_BBOX)

    model.load_weights(WEIGHTS)
    model.predict_folder(IMAGE_DIR,
                         RES_DIR,
                         apply_color_palette=APPLY_COLOR,
                         crop_bbox=CROP_BBOX_PIL, 
                         resize_shape=None)