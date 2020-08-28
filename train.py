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
    NETWORK   = 'FPNModel'                # FPNModel, UNetModel
    BACKBONE  = 'ResNet50'                # MobileNetV2, ResNet50
    LOGDIR    = './log/ms_carla_fpn_resnet_step_up22/' # where to save tensorboard logging and model weights
    N_CLASS   = 13                               # number of segmentation classes
    
    INIT_LR    = 1e-2               # initial learning rate
    DECAY      = 0.0                # learning rate decay by epoch
    LR_STEP    = [200, 400]         # epochs at which to change learning rate by LR_FACTOR
    LR_FACTOR  = 0.2                # multiplier to learning rate applied after passing an epoch in LR_STEP
    N_EPOCHS   = 500                # number of epochs to train model for
    BATCH_SIZE = 16                 # batch size
    LOG_FREQ   = 10                 # how often (epochs) to log train/val statistics
    SAVE_FREQ  = N_EPOCHS // 5      # how often (epochs) to save the intermediate weights             

    CROP_BBOX = [0, 0, 448, 800]    # start_y, start_x, delta_y, delta_x (y -> height, x -> width)

    train_imgs  = glob.glob('./train/images/*.png')
    val_imgs    = glob.glob('./val/images/*.png')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d, init_lr=%f, decay=%f, lr_step=%s, lr_factor=%f)" %
        (NETWORK, BACKBONE, N_CLASS, INIT_LR, DECAY, LR_STEP, LR_FACTOR))
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
