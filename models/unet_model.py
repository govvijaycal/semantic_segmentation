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
            layer._name = self._backbone + '/' + layer.name
            layer.trainable = False
        fmaps = [base_model.get_layer(self._backbone + '/' + fmap_name).output for fmap_name in self._feature_map_list]

        # DECODER: Learned from scratch.  Output size is related to size of feature maps
        #          in fmaps.
        seg3 = ZeroPadding2D((1, 1), name='Seg/zp11_f3')(fmaps[3])
        seg3 = Conv2D(512, (3, 3), padding='valid', activation='relu', name='Seg/conv33_seg3')(seg3)
        seg3 = BatchNormalization(name='Seg/BN_seg3')(seg3)
        seg3 = UpSampling2D((2, 2), name='Seg/up22_seg3')(seg3)

        seg2 = concatenate([seg3, fmaps[2]], axis=-1, name='Seg/concat_seg3')
        seg2 = ZeroPadding2D((1, 1), name='Seg/zp11_f2')(seg2)
        seg2 = Conv2D(256, (3, 3), padding='valid', activation='relu', name='Seg/conv33_seg2')(seg2)
        seg2 = BatchNormalization(name='Seg/BN_seg2')(seg2)
        seg2 = UpSampling2D((2, 2), name='Seg/up22_seg2')(seg2)

        seg1 = concatenate([seg2, fmaps[1]], axis=-1, name='Seg/concat_seg2')
        seg1 = ZeroPadding2D((1, 1), name='Seg/zp11_f1')(seg1)
        seg1 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='Seg/conv33_seg1')(seg1)
        seg1 = BatchNormalization(name='Seg/BN_seg1')(seg1)
        seg1 = UpSampling2D((2, 2), name='Seg/up22_seg1')(seg1)

        seg0 = concatenate([seg1, fmaps[0]], axis=-1, name='Seg/concat_seg1')
        seg0 = ZeroPadding2D((1, 1), name='Seg/zp11_f0')(seg0)
        seg0 = Conv2D(64, (3, 3), padding='valid', activation='relu', name='Seg/conv33_seg0')(seg0)
        seg0 = BatchNormalization(name='Seg/BN_seg0')(seg0)

        out = Conv2D(self._num_classes, (3, 3), padding='same', activation='softmax', name='Seg/conv33_softmax_final')(seg0)
        out = UpSampling2D((2, 2), name='Seg/up22_final')(out)

        unet_model = Model(inputs=base_model.input, outputs=out, name='unet_model')

        unet_model.compile(
            optimizer=SGD(lr=self._init_lr, momentum=0.9, nesterov=True, clipnorm=10.),
            loss=UNetModel._soft_dice_loss(cross_entropy_weight=0.1),
            metrics=[UNetModel._mean_intersection_over_union()]
        )

        return unet_model

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to run on.
    m = UNetModel(backbone = 'ResNet50')

    import time    
    import numpy as np
    import matplotlib.pyplot as plt

    image = np.random.random((4, 800, 448, 3)) * 255.0
    times = []
    
    for i in range(100):
        st = time.time()
        m._model.predict(image)
        times.append(time.time() - st)

    
    plt.plot(np.arange(100), times)
    plt.show()

    print('Mean: ', np.mean(times[5:]))
    print('Std: ',  np.std(times[5:]))
    print('Min: ',  np.min(times[5:]))
    print('Max: ',  np.max(times[5:]))

    print(m._model.summary())

    from tensorflow.keras.utils import plot_model
    plot_model(m._model, 'unet_resnet50.png')