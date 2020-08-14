""" This provides a FPN semantic segmentation model implementation.  
    This is adapted from the FPN architecture described by:
    https://arxiv.org/pdf/1901.02446.pdf
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, \
                                    ZeroPadding2D, UpSampling2D, Softmax, ReLU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.layers import GroupNormalization

# Hacky relative import to models/ directory.
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from seg_model_base import SegModelBase

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
            layer._name = self._backbone + '/' + layer.name
            layer.trainable = False
        fmaps = [base_model.get_layer(self._backbone + '/' + fmap_name).output for fmap_name in self._feature_map_list]


        # DECODER: Learned from scratch.  Output size is related to size of feature maps
        #          in fmaps.
        # Feature Pyramid Network (p4, p3, p2, p1)
        p4 = Conv2D(256, (1,1), padding='same', name='FPN/conv11_c4')(fmaps[4])        
        p3 = Conv2D(256, (1,1), padding='same', name='FPN/conv11_c3')(fmaps[3])        
        p2 = Conv2D(256, (1,1), padding='same', name='FPN/conv11_c2')(fmaps[2])        
        p1 = Conv2D(256, (1,1), padding='same', name='FPN/conv11_c1')(fmaps[1])

        p3 = p3 + UpSampling2D((2,2), name='FPN/up22_p4')(p4)
        p2 = p2 + UpSampling2D((2,2), name='FPN/up22_p3')(p3)
        p1 = p1 + UpSampling2D((2,2), name='FPN/up22_p2')(p2)

        seg4 = Conv2D(256, (3,3), padding='same', name='FPN/conv33_p4')(p4) # 1/32        
        seg3 = Conv2D(256, (3,3), padding='same', name='FPN/conv33_p3')(p3) # 1/16        
        seg2 = Conv2D(256, (3,3), padding='same', name='FPN/conv33_p2')(p2) # 1/8        
        seg1 = Conv2D(256, (3,3), padding='same', name='FPN/conv33_p1')(p1) # 1/4

        # Segmentation Part: get a set of maps of the same shape and apply final softmax on their sum.
        for ind in range(3):
            seg4 = Conv2D(128, (3,3), padding='same', activation='relu', name='Seg/conv33_seg4_%d' % ind)(seg4)
            seg4 = GroupNormalization(groups=4, name='Seg/GN_seg4_%d' % ind)(seg4)
            seg4 = UpSampling2D((2,2), interpolation='bilinear', name='Seg/up22_seg4_%d' % ind)(seg4)

        for ind in range(2):
            seg3 = Conv2D(128, (3,3), padding='same', activation='relu', name='Seg/conv33_seg3_%d' % ind)(seg3)            
            seg3 = GroupNormalization(groups=4, name='Seg/GN_seg3_%d' % ind)(seg3)            
            seg3 = UpSampling2D((2,2), interpolation='bilinear', name='Seg/up22_seg3_%d' % ind)(seg3)

        for ind in range(1):
            seg2 = Conv2D(128, (3,3), padding='same', activation='relu', name='Seg/conv33_seg2_%d' % ind)(seg2)            
            seg2 = GroupNormalization(groups=4, name='Seg/GN_seg2_%d' % ind)(seg2)
            seg2 = UpSampling2D((2,2), interpolation='bilinear', name='Seg/up22_seg2_%d' % ind)(seg2)
            
        seg1 = Conv2D(128, (3,3), padding='same', activation='relu', name='Seg/conv33_seg1')(seg1)
        seg1 = GroupNormalization(groups=4, name='Seg/GN_seg1')(seg1)
                
        out = Conv2D(self._num_classes, (1, 1), padding='same', name='Seg/conv11_final')(seg1 + seg2 + seg3 + seg4)        
        out = UpSampling2D((4, 4), interpolation='bilinear', name='Seg/up44_final')(out)
        out = Softmax(name='Seg/softmax_final')(out)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose which GPU to run on.
    m = FPNModel(backbone = 'ResNet50')

    import time    
    import numpy as np
    import matplotlib.pyplot as plt

    image = np.random.random((4, 800, 448, 3)) * 255.0
    times = []
    
    for i in range(100):
        st = time.time()
        m._model.predict(image)
        times.append(time.time() - st)

    print('Mean: ', np.mean(times[5:]))
    print('Std: ',  np.std(times[5:]))
    print('Min: ',  np.min(times[5:]))
    print('Max: ',  np.max(times[5:]))

    plt.plot(np.arange(100), times)
    plt.show()

    # from tensorflow.keras.utils import plot_model
    # print(m._model.summary())
    # plot_model(m._model, 'fpn_resnet50.png')