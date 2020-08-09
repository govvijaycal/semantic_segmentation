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
    LOGDIR    = './log/carla_fpn/' # where to save tensorboard logging and model weights
    N_CLASS   = 13                 # number of segmentation classes
    
    INIT_LR    = 5e-2               # initial learning rate
    DECAY      = 5e-4               # learning rate decay by epoch
    N_EPOCHS   = 500                # number of epochs to train model for
    BATCH_SIZE = 8                  # batch size
    LOG_FREQ   = 10                 # how often (epochs) to log train/val statistics
    SAVE_FREQ  = N_EPOCHS // 5      # how often (epochs) to save the intermediate weights             

    CROP_BBOX = [0, 0, 448, 800]    # start_y, start_x, delta_y, delta_x (y -> height, x -> width)

    train_imgs  = glob.glob('./train/images/*.png')
    val_imgs    = glob.glob('./val/images/*.png')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d, init_lr=%f, decay=%f)" %
        (NETWORK, BACKBONE, N_CLASS, INIT_LR, DECAY))
    parse_fun = partial(dataloader.parse_image, num_seg_classes=N_CLASS, crop_bbox=CROP_BBOX)
    
    model.fit_model(train_imgs,
                    val_imgs,
                    parse_function=parse_fun,
                    tf_augment_function=dataloader.tf_train_function,
                    tf_val_augment_function=dataloader.tf_val_function,
                    logdir=LOGDIR,
                    num_epochs=N_EPOCHS+1, # hack: add one so range will hit N_EPOCHS
                    batch_size=BATCH_SIZE,
                    log_epoch_freq=LOG_FREQ,
                    save_epoch_freq=SAVE_FREQ)