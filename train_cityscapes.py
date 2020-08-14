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
    NETWORK   = 'FPNModel'                         # FPNModel, UNetModel
    BACKBONE  = 'ResNet50'                         # MobileNetv2, ResNet50
    LOGDIR    = './log/city_fpn_resnet/'           # where to save tensorboard logging and model weights
    N_CLASS   = 20                                 # number of segmentation classes
    
    INIT_LR    = 5e-2               # initial learning rate
    DECAY      = 5e-4               # learning rate decay by epoch
    N_EPOCHS   = 1500               # number of epochs to train model for
    BATCH_SIZE = 16                 # batch size
    LOG_FREQ   = 10                 # how often (epochs) to log train/val statistics
    SAVE_FREQ  = N_EPOCHS // 5      # how often (epochs) to save the intermediate weights             

    TRAIN_DIR = '/shared/govvijay/cityscapes/leftImg8bit/train/'
    VAL_DIR   = '/shared/govvijay/cityscapes/leftImg8bit/val/'
    
    train_imgs = glob.glob(TRAIN_DIR + '*/*.png')
    val_imgs   = glob.glob(VAL_DIR + '*/*.png')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to run on.
    
    ##########################################################
    ##########################################################
    model = eval("%s(backbone='%s', num_classes=%d, init_lr=%f, decay=%f)" %
        (NETWORK, BACKBONE, N_CLASS, INIT_LR, DECAY))    
    
    model.fit_model(train_imgs,
                    val_imgs,
                    parse_function=dataloader.parse_image_cityscapes,
                    tf_augment_function=dataloader.tf_train_function,
                    tf_val_augment_function=dataloader.tf_val_function,
                    logdir=LOGDIR,
                    num_epochs=N_EPOCHS+1, # hack: add one so range will hit N_EPOCHS
                    batch_size=BATCH_SIZE,
                    log_epoch_freq=LOG_FREQ,
                    save_epoch_freq=SAVE_FREQ)





        

        