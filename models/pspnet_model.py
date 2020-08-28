""" This provides a PSPNet semantic segmentation model implementation.  
    This is based on the implementation from https://github.com/divamgupta/image-segmentation-keras.

    I found this has low performance, perhaps there is more optimization with feature map choice
    and adding capacity.  Try UNet or FPN first.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, \
                                    ZeroPadding2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam

# Hacky relative import to models/ directory.
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from seg_model_base import SegModelBase

class PSPNetModel(SegModelBase):

    def __init__(self, backbone='ResNet50', num_classes=13, init_lr=5e-2, decay=5e-4, lr_step=[], lr_factor=0.0):
        """ Constructor with adjustable base pretrained CNN model and segmentation target size.

        Args:
            backbone: Pretrained CNN backbone model used for feature extraction.
            num_classes: Number of segmentation classes to predict.
            init_lr: initial learning rate to use under the SGD with Momentum optimizer
            decay: time-based learning rate decay rate (epoch)
        """
        for key in list(locals()):
            if key == 'self':
                pass
            else:
                setattr(self, '_%s' % key, locals()[key])

        if decay > 0.0 and len(lr_step) == 0:
            print('Using learning rate decay at epoch rate: %.3f' % decay)
            self._lr_mode = 'decay'
        elif len(lr_step) > 0 and lr_factor > 0.0:
            assert lr_factor < 1.0, "lr_factor should be in (0.0, 1.0)"
            print('Using learning rate schedule with drops at %s with factor %.3f' % (lr_step, lr_factor))
            self._lr_mode = 'step'
        else:
            raise ValueError("Did not specify a valid learning rate schedule")

        # Based on backbone selection, pick out 4 feature maps used in the UNet architecture.
        if self._backbone == 'MobileNetV2':
            self._feature_map_list = ['block_1_expand_relu', 'block_3_expand_relu',
                                      'block_6_expand_relu', 'block_13_expand_relu']
        elif self._backbone == 'ResNet50':
            self._feature_map_list = ['conv1_conv', 'conv2_block3_out',
                                      'conv3_block4_out', 'conv4_block6_out']
        else:
            raise ValueError("Invalid backbone selection.")

        super().__init__()

    def _create_model(self):
        """ Model construction method called from constructor. """

        # ENCODER: Pretrained and frozen CNN model used to extract feature maps (fmaps).
        if self._backbone == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            base_model_fn = MobileNetV2
        elif self._backbone == 'ResNet50':
            from tensorflow.keras.applications.resnet50 import ResNet50
            base_model_fn = ResNet50
        else:
            raise ValueError("Invalid backbone selection.")

        # *** NOTE: This fails if the actual input_shape is not divisable by 32.
        #           We assume the user will handle this appropriately in image preprocessing.
        base_model = base_model_fn(include_top=False,
                                   weights='imagenet',
                                   input_shape=(224, 224, 3),
                                   pooling=None)
        for layer in base_model.layers:
            layer.trainable = False
        fmaps = [base_model.get_layer(fmap_name).output for fmap_name in self._feature_map_list]

        # DECODER: Learned from scratch.  Output size is related to size of feature maps
        #          in fmaps.  
        # The PSP Decoder adapted from:
        # https://github.com/divamgupta/image-segmentation-keras/blob/bb7ac1f5cbaa884cea94e56c1a2aa5ea1690ed26/keras_segmentation/models/pspnet.py#L52

        x = fmaps[1]
        _, x_height, x_width, x_channels = x.shape # 28 x 28 
        pool_factors = [1, 2, 4, 7]
        pool_outputs = [x]

        for pf in pool_factors:
            pool_size = (int(x_height / pf), int(x_width / pf))
            y = AveragePooling2D(pool_size, padding='same')(x)
            y = Conv2D(512, (1,1), padding='same', activation='relu')(y)
            y = BatchNormalization()(y)
            y = UpSampling2D(pool_size, interpolation='bilinear')(y)
            pool_outputs.append(y)

        y = concatenate(pool_outputs, axis=-1)
        y = Conv2D(512, (1,1), activation='relu')(y)
        y = BatchNormalization()(y)
        
        y = Conv2D(self._num_classes, (3, 3), padding='same', activation='softmax')(y)
        y = UpSampling2D((4, 4))(y)

        pspnet_model = Model(inputs=base_model.input, outputs=y, name='pspnet_model')

        pspnet_model.compile(
            optimizer=SGD(lr=self._init_lr, momentum=0.9, nesterov=True, clipnorm=10.),
            loss=PSPNetModel._soft_dice_loss(cross_entropy_weight=0.1),
            metrics=[PSPNetModel._mean_intersection_over_union()]
        )
        
        return pspnet_model

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.
    m = PSPNetModel(backbone = 'MobileNetV2')
