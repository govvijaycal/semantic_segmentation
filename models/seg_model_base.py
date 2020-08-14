""" Segmentation Model Base Class Implementation.
    Essentially implements all training, loss, and prediction functions,
    but leaves network architecture to subclass implementation.
"""
import glob
from datetime import datetime
from functools import partial
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
import os
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.losses import categorical_crossentropy

# Hacky relative import to ../ directory to use palette conversion utils.
import os
import sys
root_dir = os.path.dirname( os.path.dirname(os.path.abspath(__file__)) )
sys.path.insert(0, root_dir)
from utils.apply_color_palette import convert_label_to_palette_img

class SegModelBase(ABC):
    
    def __init__(self):
        """ Constructor.  Details of the model architecture are to be implemented by subclasses. """
        
        if hasattr(self, '_decay'):
            pass
        else:
            self._decay = 0.0

        self._trained = False
        self._model = self._create_model()

    @abstractmethod
    def _create_model(self):
        """ Model architecture defined by subclass.  Creates self._model, a Keras compiled model. """
        pass

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
            return tf.reduce_mean((intersection + epsilon) / (union + epsilon)) # mean over batch and channels

        return metric

    @staticmethod
    def _soft_dice_loss(epsilon=1e-10, cross_entropy_weight=0.0):
        """ Soft Dice Loss based on https://www.jeremyjordan.me/semantic-segmentation/#loss.
        Loss function reference: https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
        """
        def loss(y_true, y_pred):
            # y_true: batch x height x width x channels (one-hot encoding of binary label)
            # y_pred: batch x height x width x channels (e.g. softmax output)
            sum_axes = tuple(range(1, len(y_pred.shape) - 1))  # height, width axes
            num = 2. * tf.reduce_sum(y_pred * y_true, sum_axes)  # batch x channels
            den = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true), sum_axes)  # batch x channels

            dice_loss = 1 - tf.reduce_mean((num + epsilon) / (den + epsilon))  # mean over batch and channels
            if cross_entropy_weight > 0.0:
                dice_loss += cross_entropy_weight * tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
            return dice_loss

        return loss

    @staticmethod
    def _calc_cyclic_val(epoch, min_val, max_val, period, flip=False):
        # triangular function of epoch oscillating from min_val to max_val
        # flip = True  => start from min_val at beginning of period
        # flip = False => start from max_val at beginning of period
        epoch_mod = epoch % period
        half_period = period / 2.
        slope = (max_val - min_val) / half_period
        if epoch_mod < half_period:
            val = min_val + slope * epoch_mod
        else:
            val = max_val - slope * (epoch_mod - half_period)
    
        if flip:
            val = max_val - val + min_val
        return val
        # To use this for cyclic learning rate and momentum:
        # next_lr       = SegModelBase._calc_cyclic_val(epoch=epoch+1, min_val=init_lr/10., max_val=init_lr, period=10.)
        # next_momentum = SegModelBase._calc_cyclic_val(epoch=epoch+1, min_val=0.8, max_val=0.9, period=10., flip=True)
        # K.set_value(self._model.optimizer.lr, next_lr)
        # K.set_value(self._model.optimizer.momentum, next_momentum)

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
        val_dataset = val_dataset.batch(2) # batch size not critical here
        #val_dataset = val_dataset.prefetch(buffer_size=2*batch_size)

        init_lr = K.get_value(self._model.optimizer.lr)
        
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=False)
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

            next_lr = init_lr / (1 + self._decay * epoch)
            K.set_value(self._model.optimizer.lr, next_lr)

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

    def predict_folder(self, image_dir, res_dir, color_palette=None, crop_bbox=None, resize_shape=None):
        """ Prediction applied to a full image_dir and then saved in res_dir.
        This function simply runs predictions on all images in image_dir and saves to res_dir.
        The argmax is taken, so the results are a grayscale image and not the full softmax output.
        It does not do any preprocessing (e.g. cropping) except for image resizing.
        It also does not do evaluation, as labels are not considered even if available.

        Args:
            image_dir: location with images to make predictions on
            res_dir: location to save prediction result images             
            color_palette: convert label values with this color palette (see utils/apply_color_palette.py)
            crop_bbox: None or [x_min, y_min, x_max, y_max].
            resize_shape: None or [width, height]
        """
        
        os.makedirs(res_dir, exist_ok=True)

        img_list = glob.glob(image_dir + '**/*.png', recursive=True)
        img_list.extend(glob.glob(image_dir + '**/*.jpg', recursive=True)) # both png/jpg supported

        print('Predicting from folder:')
        for imagepath in img_list:
            # While this could be done fully in tf, I want to make sure it can be
            # deployed with numpy, e.g. for real-time image acquisition.
            img_raw = Image.open(imagepath).convert('RGB')
            if crop_bbox:
                img_raw = img_raw.crop(box=tuple(crop_bbox))
            if resize_shape:
                img_raw = img_raw.resize(tuple(resize_shape))
            img_raw = np.array(img_raw)

            seg_pred = self.predict_instance(img_raw)
            savepath = res_dir + imagepath.split('/')[-1]

            print('  imagepath: ', imagepath)
            print('  savepath: ', savepath)

            label_img = np.argmax(seg_pred, axis=-1).astype(np.uint8)

            if color_palette:
                pal_img = convert_label_to_palette_img(label_img, color_palette)
                Image.fromarray(pal_img).save(savepath)
            else:
                Image.fromarray(label_img).save(savepath)