""" Utility functions for loading image/seg image pairs and performing augmentation.
    This is tuned for a Carla dataset (13 classes) with 800x600 images with crop
    to a 224x224 augmented image.
"""
import glob
from functools import partial
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa


def parse_image(img_path, num_seg_classes=13, crop_bbox=None):
    """ Takes an image directory and returns dictionary of images and labels.

    Please note that current implementation assumes the image path ends with "images"
    and that the label path ends with "labels".  The filename in image_path
    should match the filename in the label path.

    Args:
        img_path: Path to directory containing images (assumed .png).
        num_seg_classes: Number of segmentation classes for one-hot encoding.
        crop_bbox: [start_x, start_y, delta_x, delta_y] to crop image if not None

    Returns:
        A dictionary with key "image" corresponding to loaded image,
        "label" corresponding to the segmentation mask with one-hot encoding,
        and "name" giving the filename without path.


    Reference: https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
    """
    name = tf.strings.split(img_path, sep='/')[-1]
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    if crop_bbox is not None:
        image = tf.image.crop_to_bounding_box(image, crop_bbox[0], crop_bbox[1],
                                              crop_bbox[2], crop_bbox[3])

    mask_path = tf.strings.regex_replace(img_path, "images", "labels") # assumed directory structure
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    if crop_bbox is not None:
        mask = tf.image.crop_to_bounding_box(mask, crop_bbox[0], crop_bbox[1],
                                             crop_bbox[2], crop_bbox[3])

    mask = tf.one_hot(tf.squeeze(mask), num_seg_classes, dtype=tf.uint8) # H x W -> H x W x channels

    return {'image': image, 'label': mask, 'name': name}


def augment_function(image, label):
    """ Takes in an image, segmentation label pair and returns augmented variants for training.
    Note that geometric augmentations are done first to rescale/flip and get a 224x224 image.
    Then non-geometric augmentions are done to perturb the 224x224 image for training.

    # Reference: https://github.com/divamgupta/image-segmentation-keras/blob/bb7ac1f5cbaa884cea94e56c1a2aa5ea1690ed26/keras_segmentation/data_utils/augmentation.py#L77
     """
    sometimes = lambda aug: iaa.Sometimes(.75, aug)  # apply the augmentation 75% of the time
    seq = iaa.Sequential([
        # Step 1: Resize to (224, 224) either by direct rescaling or random scale then crop.
        iaa.OneOf([
            iaa.Resize(224),
            iaa.Sequential([
                iaa.Resize({"height": (0.5, 0.75), "width": (0.3, 0.75)}),
                iaa.CropToFixedSize(width=224, height=224),
                ], random_order=False)
        ]),
        # Step 2: Flip horizontally with 50% chance.
        iaa.Fliplr(0.5),
        # Step 3: Apply 0 - 3 of following transforms, where each transform is applied sometimes
        #         and in random order.
        iaa.SomeOf((0, 3), [
            sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
            sometimes(iaa.LinearContrast((0.5, 1.5))),
            sometimes(iaa.MultiplyBrightness((0.75, 1.25))),
            sometimes(iaa.MultiplySaturation((0.5, 1.5))),
            sometimes(iaa.Rain(speed=(0.1, 0.3))),
            sometimes(iaa.Clouds())
        ], random_order=True),
        # Step 4: Randomly drop out 0 - 20% of pixels, sometimes.
        sometimes(iaa.Dropout(p=(0, 0.2)))
    ], random_order=False)

    image_aug, label_aug = seq(images=image, segmentation_maps=label)  # image_aug is a list
    return np.array(image_aug), np.array(label_aug)


def val_augment_function(image, label):
    """ Takes in an image, segmentation label and returns augmented variants for validation.

    In particular, this implementation just resizes to the right size.
    """
    seq = iaa.Resize(224)  # 224 x 224 size for both image and label (not keeping aspect ratio)
    image_val, label_val = seq(images=image, segmentation_maps=label) # image_val is a list
    return np.array(image_val), np.array(label_val)

def tf_augment_function(instance_dict):
    """ Helper function to wrap imgaug training augmentation into a tf numpy_function """
    image_shape = [None, 224, 224, 3]
    label_shape = [None, 224, 224, 13]
    [image, label] = tf.numpy_function(
        augment_function, [instance_dict["image"], instance_dict["label"]],
        [tf.uint8, tf.uint8])
    image.set_shape(image_shape)
    label.set_shape(label_shape)

    return image, label

def tf_val_augment_function(instance_dict):
    """ Helper function to wrap imgaug validation augmentation into a tf numpy_function """
    image_shape = [None, 224, 224, 3]
    label_shape = [None, 224, 224, 13]
    [image, label] = tf.numpy_function(
        val_augment_function,
        [instance_dict["image"], instance_dict["label"]],
        [tf.uint8, tf.uint8])
    image.set_shape(image_shape)
    label.set_shape(label_shape)

    return image, label

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.

    import matplotlib.pyplot as plt

    TRAIN_DIR = './train/'
    VAL_DIR = './val/'

    np.random.seed(0)
    train_imgs = np.array(glob.glob(TRAIN_DIR + 'images/*.png'))
    val_imgs = np.array(glob.glob(VAL_DIR + 'images/*.png'))
    np.random.shuffle(train_imgs)
    np.random.shuffle(val_imgs)

    parse_function = partial(parse_image, num_seg_classes=13, crop_bbox=[0, 0, 450, 800])

    train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
    train_dataset = train_dataset.shuffle(5, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.batch(16)
    train_dataset = train_dataset.map(tf_augment_function,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    for ind, (img_aug, seg_aug) in enumerate(train_dataset):
        # Preview some of the loaded images and labels for debugging.
        # Number of images shown is arbitrary, adjust as needed.
        plt.subplot(321)
        plt.imshow(img_aug[0])
        plt.subplot(322)
        plt.imshow(np.argmax(seg_aug[0], axis=-1), cmap='jet')

        plt.subplot(323)
        plt.imshow(img_aug[1])
        plt.subplot(324)
        plt.imshow(np.argmax(seg_aug[1], axis=-1), cmap='jet')

        plt.subplot(325)
        plt.imshow(img_aug[2])
        plt.subplot(326)
        plt.imshow(np.argmax(seg_aug[2], axis=-1), cmap='jet')

        plt.show()

        if ind > 2:
            break
