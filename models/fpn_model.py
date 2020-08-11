""" This provides a FPN semantic segmentation model implementation.  
    This is adapted from the FPN architecture described by:
    https://arxiv.org/pdf/1901.02446.pdf
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, \
                                    ZeroPadding2D, UpSampling2D, Softmax, ReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.layers import GroupNormalization

from .seg_model_base import SegModelBase

class FPNModel(SegModelBase):

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

        # Based on backbone selection, pick out 5 feature maps to use in the FPN.
        # Like in the Panoptic FPN paper, we drop the 1/2 resolution.
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
        # Feature Pyramid Network (p4, p3, p2, p1)
        p4 = Conv2D(256, (1,1), padding='same')(fmaps[4])
        p3 = Conv2D(256, (1,1), padding='same')(fmaps[3])
        p2 = Conv2D(256, (1,1), padding='same')(fmaps[2])
        p1 = Conv2D(256, (1,1), padding='same')(fmaps[1])

        p3 = p3 + UpSampling2D((2,2))(p4)
        p2 = p2 + UpSampling2D((2,2))(p3)
        p1 = p1 + UpSampling2D((2,2))(p2)

        seg4 = Conv2D(256, (3,3), padding='same')(p4) # 1/32
        seg3 = Conv2D(256, (3,3), padding='same')(p3) # 1/16
        seg2 = Conv2D(256, (3,3), padding='same')(p2) # 1/8
        seg1 = Conv2D(256, (3,3), padding='same')(p1) # 1/4

        # Segmentation Part: get a set of maps of the same shape and apply final softmax on their sum.
        for _ in range(3):
            seg4 = Conv2D(64, (3,3), padding='same')(seg4)
            seg4 = GroupNormalization(groups=4)(seg4)
            seg4 = ReLU()(seg4)
            seg4 = UpSampling2D((2,2), interpolation='bilinear')(seg4)

        for _ in range(2):
            seg3 = Conv2D(64, (3,3), padding='same')(seg3)
            seg3 = GroupNormalization(groups=4)(seg3)
            seg3 = ReLU()(seg3)
            seg3 = UpSampling2D((2,2), interpolation='bilinear')(seg3)

        for _ in range(1):
            seg2 = Conv2D(64, (3,3), padding='same')(seg2)
            seg2 = GroupNormalization(groups=4)(seg2)
            seg2 = ReLU()(seg2)
            seg2 = UpSampling2D((2,2), interpolation='bilinear')(seg2)
            
        seg1 = Conv2D(64, (3,3), padding='same')(seg1)
        seg1 = GroupNormalization(groups=4)(seg1)
        seg1 = ReLU()(seg1)
        
        out = Conv2D(self._num_classes, (1, 1), padding='same')(seg1 + seg2 + seg3 + seg4)
        out = UpSampling2D((4, 4), interpolation='bilinear')(out)
        out = Softmax()(out)

        fpn_model = Model(inputs=base_model.input, outputs=out, name='fpn_model')

        fpn_model.compile(
            optimizer=SGD(lr=self._init_lr, momentum=0.9, nesterov=True, clipnorm=10.),
            loss=FPNModel._soft_dice_loss(),
            metrics=[FPNModel._mean_intersection_over_union()]
        )
        
        return fpn_model

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.
    m = FPNModel(backbone = 'ResNet50')

    from tensorflow.keras.utils import plot_model
    plot_model(m._model, 'fpn_model.png')
    print(m._model.summary())