""" This provides a UNet semantic segmentation model implementation.  
    This is based on the implementation from https://github.com/divamgupta/image-segmentation-keras.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, \
                                    ZeroPadding2D, UpSampling2D
from tensorflow.keras.optimizers import SGD, Adam

# Hacky relative import to models/ directory.
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from seg_model_base import SegModelBase

class UNetModel(SegModelBase):

    def __init__(self, backbone='ResNet50', num_classes=13, init_lr=5e-2, decay=5e-4):
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

        # Based on backbone selection, pick out 5 feature maps to use for the encoder in UNet.
        # Current implementation drops the 1/32 resolution feature map but could be added later.
        if self._backbone == 'MobileNetV2':
            self._feature_map_list = ['block_1_expand_relu',    # 1/2
                                      'block_3_expand_relu',    # 1/4
                                      'block_6_expand_relu',    # 1/8 
                                      'block_13_expand_relu',   # 1/16
                                      'block_16_depthwise_relu' # 1/32
                                     ]
        elif self._backbone == 'ResNet50':
            self._feature_map_list = ['conv1_relu',             # 1/2
                                      'conv2_block3_out',       # 1/4
                                      'conv3_block4_out',       # 1/8
                                      'conv4_block6_out',       # 1/16
                                      'conv5_block3_out'        # 1/32
                                     ]
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
                                   input_shape=(None, None, 3),
                                   pooling=None)
        for layer in base_model.layers:
            layer.trainable = False
        fmaps = [base_model.get_layer(fmap_name).output for fmap_name in self._feature_map_list]

        # DECODER: Learned from scratch.  Output size is related to size of feature maps
        #          in fmaps.
        x = ZeroPadding2D((1, 1))(fmaps[3])
        x = Conv2D(512, (3, 3), padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, fmaps[2]], axis=-1)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(256, (3, 3), padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, fmaps[1]], axis=-1)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3), padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, fmaps[0]], axis=-1)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, (3, 3), padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(self._num_classes, (3, 3), padding='same', activation='softmax')(x)
        x = UpSampling2D((2, 2))(x)

        unet_model = Model(inputs=base_model.input, outputs=x, name='unet_model')

        unet_model.compile(
            optimizer=SGD(lr=self._init_lr, momentum=0.9, nesterov=True, clipnorm=10.),
            loss=UNetModel._soft_dice_loss(),
            metrics=[UNetModel._mean_intersection_over_union()]
        )

        return unet_model

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.
    m = UNetModel(backbone = 'ResNet50')