""" This provides a UNet semantic segmentation model implementation, tuned for settings with
    a CARLA dataset (13 classes) and 224 x 224 image input.  This is based on the implementation
    from https://github.com/divamgupta/image-segmentation-keras.
"""
import glob
from datetime import datetime
from functools import partial
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, \
                                    ZeroPadding2D, UpSampling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import TensorBoard

import utils.dataloader as dl

import pdb

class PSPNetModel():
    '''
        UNet Model, adapted from https://github.com/divamgupta/image-segmentation-keras.

        This is defaulting to settings in CARLA, which has 13 semantic segmentation classes.
        https://carla.readthedocs.io/en/stable/cameras_and_sensors/#camera-semantic-segmentation
    '''
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

        # Based on backbone selection, pick out 4 feature maps used in the UNet architecture.
        if self._backbone == 'MobileNetV2':
            self._feature_map_list = ['block_1_expand_relu', 'block_3_expand_relu',
                                      'block_6_expand_relu', 'block_13_expand_relu']
        elif self._backbone == 'ResNet50':
            self._feature_map_list = ['conv1_conv', 'conv2_block3_out',
                                      'conv3_block4_out', 'conv4_block6_out']
        else:
            raise ValueError("Invalid backbone selection.")

        self._trained = False
        self._model = self._create_model()

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

        base_model = base_model_fn(include_top=False,
                                   weights='imagenet',
                                   input_shape=(224, 224, 3),
                                   pooling=None)
        for layer in base_model.layers:
            layer.trainable = False
        fmaps = [base_model.get_layer(fmap_name).output for fmap_name in self._feature_map_list]

        # DECODER: Learned from scratch.  Output size is related to size of feature maps
        #          in fmaps.  The PSP Decoder adapted from:
        # https://github.com/divamgupta/image-segmentation-keras/blob/bb7ac1f5cbaa884cea94e56c1a2aa5ea1690ed26/keras_segmentation/models/pspnet.py#L52

        x = fmaps[2]
        _, x_height, x_width, x_channels = x.shape  
        pool_size = [1, 2, 4, 8]
        pool_outputs = [x]

        for pf in pool_factors:
            pool_size = (pf, pf)
            y = AveragePooling2D(pool_size, padding='same')(x)
            y = Conv2D(512, (1,1), padding='same', activation='relu')(y)
            y = BatchNormalization()(y)
            pool_outputs.append(y)

        pdb.set_trace()



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
            #optimizer=SGD(lr=self._init_lr, momentum=0.9, nesterov=True, clipnorm=10.),
            optimizer=Adam(lr=self._init_lr, clipnorm=10.),
            loss=UNetModel._soft_dice_loss(),
            metrics=[UNetModel._mean_intersection_over_union()]
        )

        print(unet_model.summary())
        return unet_model

    @staticmethod
    def _mean_intersection_over_union(epsilon=1e-10):
        """ MIoU based on https://www.jeremyjordan.me/evaluating-image-segmentation-models/ """

        def metric(y_true, y_pred):
            # y_true: batch x height x width x channels (one-hot encoding of binary label)
            # y_pred: batch x height x width x channels (e.g. softmax output)
            y_true_mask = tf.cast(y_true, tf.float32)
            y_pred_mask = tf.one_hot(tf.argmax(y_pred, axis=-1), y_pred.shape[-1], dtype=tf.float32)

            sum_axes = tuple(range(1, len(y_pred.shape)-1))  # height, width axes

            # Both intersection and union tensors have shape batch x channels.
            intersection = tf.reduce_sum(y_true_mask * y_pred_mask, sum_axes)
            union = tf.reduce_sum(tf.clip_by_value(y_true_mask + y_pred_mask, 0, 1), sum_axes)
            return tf.reduce_mean((intersection) / (union + epsilon)) # mean over batch and channels

        return metric

    @staticmethod
    def _soft_dice_loss(epsilon=1e-10):
        """ Soft Dice Loss based on https://www.jeremyjordan.me/semantic-segmentation/#loss.
        Loss function reference: https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
        """
        def loss(y_true, y_pred):
            # y_true: batch x height x width x channels (one-hot encoding of binary label)
            # y_pred: batch x height x width x channels (e.g. softmax output)
            sum_axes = tuple(range(1, len(y_pred.shape) - 1))  # height, width axes
            num = 2. * tf.reduce_sum(y_pred * y_true, sum_axes)  # batch x channels
            den = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), sum_axes)  # batch x channels

            return 1 - tf.reduce_mean((num) / (den + epsilon))  # mean over batch and channels

        return loss

    def fit_model(self,
                  train_image_list,
                  val_img_list,
                  parse_function=None,
                  tf_augment_function=None,
                  tf_val_augment_function=None,
                  logdir=None,
                  num_epochs=100,
                  batch_size=16,
                  log_epoch_freq=10,
                  save_epoch_freq=20):
        """ Main training function.

        Args:
            train_image_list: List of training images to load and augment.
            val_img_list: List of validation images to load and augment.
            parse_function: Function to load image/segmentation mask pairs.
            tf_augment_function: Augmentation function used for training.
            tf_val_augment_function: Augmentation function used for validation.
            logdir: Where intermediate saved weights and tensorboard logging are saved.
            num_epochs: How many epochs to train for.
            batch_size: Batch size used for training.
            log_epoch_freq: How often to evaluate train/validation loss metrics for Tensorboard.
            save_epoch_freq: How often to save intermediate weights.
        """

        if parse_function is None:
            raise ValueError("Require parse_function for dataset image/label loading.")
        if tf_augment_function is None or tf_val_augment_function is None:
            raise ValueError("Require both augment_function and val_augment_function.  "
                             "Can provide an identity lambda function for no augmentation.")
        if logdir is None:
            raise ValueError("Need to provide a logdir for TensorBoard logging and model saving.")
        os.makedirs(logdir, exist_ok=True)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_image_list)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_image_list),
                                              reshuffle_each_iteration=True)
        train_dataset = train_dataset.map(parse_function)
        train_dataset = train_dataset.map(tf_augment_function)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=2*batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(val_img_list)
        val_dataset = val_dataset.map(parse_function)
        val_dataset = val_dataset.map(tf_val_augment_function)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(buffer_size=2*batch_size)

        init_lr = K.get_value(self._model.optimizer.lr)
        next_lr = init_lr

        tensorboard = TensorBoard(log_dir=logdir, profile_batch='5,10', histogram_freq=0, write_graph=False)
        tensorboard.set_model(self._model)

        for epoch in range(num_epochs):
            print('Epoch %d started at %s' % (epoch, datetime.now()))

            losses = []
            mious = []
            for images, labels in train_dataset:
                images = tf.cast(images, tf.float32) / 127.5 - 1.0
                loss, miou = self._model.train_on_batch(images, labels)
                losses.append(loss)
                mious.append(miou)
            epoch_loss = np.mean(losses)
            epoch_miou = np.mean(mious)
            print('\tTrain loop done at %s' % datetime.now())

            # next_lr = init_lr / (1 + self._decay * epoch)
            # K.set_value(self._model.optimizer.lr, next_lr)

            if log_epoch_freq and epoch % log_epoch_freq == 0:
                print('\tComputing Validation at %s' % datetime.now())

                # compute validation loss, miou
                val_losses = []
                val_mious = []
                for images, labels in val_dataset:
                    images = tf.cast(images, tf.float32) / 127.5 - 1.0
                    loss, miou = self._model.test_on_batch(images, labels)
                    val_losses.append(loss)
                    val_mious.append(miou)
                val_epoch_loss = np.mean(val_losses)
                val_epoch_miou = np.mean(val_mious)
                print('\tVal loop done at %s' % datetime.now())
                print('\tEpoch %d: loss %.4f MIoU %.4f val_loss %.4f val_MIoU %.4f' %
                      (epoch, epoch_loss, epoch_miou, val_epoch_loss, val_epoch_miou))

                tensorboard.on_epoch_end(epoch,
                                         {'lr': next_lr, 'loss': epoch_loss, 'MIoU' : epoch_miou,
                                          'val_loss': val_epoch_loss, 'val_MIoU' : val_epoch_miou})

            if save_epoch_freq and epoch % save_epoch_freq == 0:
                filename = logdir + 'seg_pred_%05d_epochs' % epoch
                self.save_weights(filename)
                print('\tSaving model at epoch %d to %s.' % (epoch, filename))

        tensorboard.on_train_end(None)
        self._trained = True

    def save_weights(self, filepath):
        """ Save intermediate weights, not full model. """
        self._model.save_weights('%s.h5' % filepath)

    def load_weights(self, filepath):
        """ Load weights without needing to reload full model. """
        self._model.load_weights('%s.h5' % filepath)

    def save_model(self, modeldir):
        """ Save the full model so it is standalone.
        Note that this is primarily meant to save a final model for deployment.
        """
        self._model.save(modeldir)

    def predict_instance(self, img_raw):
        """ Given an img_raw, e.g. a numpy RGB array, return softmax prediction output array.
            Assumption is input is 1 x H x W X 3 and output is H x W x num_channels.
        """
        if len(img_raw.shape) == 3:
            img_raw = np.expand_dims(img_raw, 0)  # add batch dimension
        img = img_raw.astype(np.float32) / 127.5 - 1.0
        return np.squeeze(self._model.predict(img))  # remove batch dimension

    def predict_folder(self, image_dir, res_dir):
        """ Prediction applied to a full image_dir and then saved in res_dir.
        This function simply runs predictions on all images in image_dir and saves to res_dir.
        The argmax is taken, so the results are a grayscale image and not the full softmax output.
        It does not do any preprocessing (e.g. cropping) except for image resizing.
        It also does not do evaluation, as labels are not considered even if available.
        """
        os.makedirs(res_dir, exist_ok=True)

        img_list = glob.glob(image_dir + '*.png')
        img_list.extend(glob.glob(image_dir + '*.jpg')) # both png/jpg supported

        print('Predicting from folder:')
        for imagepath in img_list:
            # While this could be done fully in tf, I want to make sure it can be
            # deployed with numpy, e.g. for real-time image acquisition.
            img_raw = Image.open(imagepath).convert('RGB')
            img_raw = np.array(img_raw.resize((224, 224)))

            seg_pred = self.predict_instance(img_raw)
            savepath = imagepath.replace(image_dir, res_dir)

            print('  imagepath: ', imagepath)
            print('  savepath: ', savepath)

            label_img = np.argmax(seg_pred, axis=-1).astype(np.uint8)
            Image.fromarray(label_img).save(savepath)

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # choose which GPU to run on.

    model = PSPNetModel(backbone='ResNet50', num_classes=13, init_lr=5e-2, decay=5e-4)
    
    """
    parse_fun = partial(dl.parse_image, num_seg_classes=13, crop_bbox=[0, 0, 450, 800])

    TRAINDIR = './train/images/'
    train_imgs = glob.glob(TRAINDIR + '*.png')
    VALDIR = './val/images/'
    val_imgs = glob.glob(VALDIR + '*.png')

    
    model.fit_model(train_imgs,
                    val_imgs,
                    parse_function=parse_fun,
                    tf_augment_function=dl.tf_train_function,
                    tf_val_augment_function=dl.tf_val_function,
                    logdir='./log/carla_psp_mobilenetv2/',
                    num_epochs=501,
                    batch_size=16,
                    log_epoch_freq=10,
                    save_epoch_freq=500)
    

    # # PREDICTIONS
    # model.load_weights('./log/carla_psp_mobilenetv2/seg_pred_00500_epochs')
    # model.predict_folder('./val/images/', './val/preds_psp_500/')
    """