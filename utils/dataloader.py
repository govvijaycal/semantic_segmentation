""" Utility functions for loading image/seg image pairs and performing augmentation.
    This is tuned for a Carla (13 classes) with 800x600 images cropped to 800x448.
"""
import glob
from functools import partial
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
#import imgaug.augmenters as iaa
import pdb

def parse_image(img_path, num_seg_classes=13, crop_bbox=None):
    """ Takes an image directory and returns dictionary of images and labels.

    Please note that current implementation assumes the image path ends with "images"
    and that the label path ends with "labels".  The filename in image_path
    should match the filename in the label path.

    Args:
        img_path: Path to directory containing images (assumed .png).
        num_seg_classes: Number of segmentation classes for one-hot encoding.
        crop_bbox: [start_y, start_x, delta_y, delta_x] to crop image if not None

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

def parse_image_cityscapes(img_path):
    """ Similar to parse_image but specialized for Cityscapes """

    #### Mask IDS
    # Note: ids 0 - 18 are valid classes to predict for (so 19 default training classes).
    # We choose to group all "ignore" or "background" pixels into a trainID 19 (so we use 20 total training classes).
    num_seg_classes_with_background = 20
    background_final_label = tf.constant(num_seg_classes_with_background - 1, dtype=tf.int32)

    ### Assumed form of img_path and directory structure.
    # We assume that img_path is of the form leftImg8bit/x/y/y_z_w_leftImg8bit.png 
    # so that the mask path is of the form   gtFine/x/y/y_z_w_gtFine_labelTrainIds.png
    # e.g. leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
    #      gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png

    name = tf.strings.split(img_path, sep='/')[-1]
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # img_shape = tf.shape(image)
    # image_height = img_shape[0]
    # image_width  = img_shape[1]
    # target_size = (image_height / 2, image_width / 2)
    target_size = (int(448), int(896))
    image = tf.image.resize(image, target_size, method='bilinear') # downsample  to [448, 896, 3]
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    
    mask_path = tf.strings.regex_replace(img_path, "leftImg8bit", "gtFine") # assumed directory structure
    mask_path = tf.strings.regex_replace(mask_path, ".png", "_labelTrainIds.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)

    mask = tf.image.resize(mask, target_size, method='nearest') # downsample by 2x to [448, 896, 1]      

    # Hack to convert label value 255 -> background_final_label = 19
    mask = tf.cast(mask, tf.int32)
    mask = tf.minimum(mask, background_final_label)                 

    mask = tf.one_hot(tf.squeeze(mask), num_seg_classes_with_background, dtype=tf.uint8) # H x W -> H x W x channels

    return {'image': image, 'label': mask, 'name': name}

def tf_train_function(instance_dict):
    image_train = tf.image.convert_image_dtype(instance_dict['image'], dtype=tf.float32)
    label_train = tf.image.convert_image_dtype(instance_dict['label'], dtype=tf.float32)

    sample_rvs = tf.random.uniform(shape=[4])

    # Step 1: For all images, resize to (224, 224).  This is not ideal as there will be
    # label noise for thin objects only resolvable at higher resolutions, but is okay
    # for essentially lane/driveable area estimation focus.  Also, this will make the 
    # trained model not robust to multi-scale/varying pixel resolution.
    image_train = tf.image.resize(image_train, (224,224), method='bilinear')
    label_train = tf.image.resize(label_train, (224,224), method='nearest')    

    img_shape = tf.shape(image_train)
    image_height = tf.cast(img_shape[0], dtype=tf.float32)
    image_width  = tf.cast(img_shape[1], dtype=tf.float32)


    # Step 2: Randomly flip left/right
    if sample_rvs[1] > 0.4:
        image_train = tf.image.flip_left_right(image_train)
        label_train = tf.image.flip_left_right(label_train)

    # Step 3: Randomly adjust brightness, contrast, and/or saturation.
    if sample_rvs[2] > 0.4:
        num_transforms = 3
        image_transform_rvs   = tf.random.uniform(shape=[num_transforms])
        image_transform_order = tf.random.shuffle(tf.range(num_transforms)) 

        for image_transform_index in image_transform_order:
            if image_transform_rvs[image_transform_index] > 0.5:
                if image_transform_index == 0:
                    image_train = tf.image.random_brightness(image_train, 0.1)      # randomly add a bias to all pixels
                elif image_transform_index == 1:
                    image_train = tf.image.random_contrast(image_train, 0.5, 2.0)   # randomly scales deviation from mean
                else:
                    image_train = tf.image.random_saturation(image_train, 0.5, 2.0) # randomly scale saturation in HSV space
    
    # Step 4: Randomly dropout a 20 x 20 patch (cutout) (< 1% of full image) to mimic occlusions/shadows.
    if sample_rvs[3] > 0.4:
        image_train = tf.expand_dims(image_train, axis=0)
        image_train = tfa.image.random_cutout(image_train, mask_size=(20,20), constant_values=0)
        image_train = tf.squeeze(image_train)

        # Note: Pixel dropout of 10% not used, left for reference.
        # mask = tf.cast( tf.random.uniform(shape=[image_height, image_width]) > 0.1, tf.float32)
        # mask_3 = tf.stack((mask,mask,mask), axis=-1)
        # image_train = image_train * mask_3

    image_train = tf.image.convert_image_dtype(image_train, dtype=tf.uint8, saturate=True)
    label_train = tf.image.convert_image_dtype(label_train, dtype=tf.uint8, saturate=True)

    return image_train, label_train

def tf_val_function(instance_dict):
    image_val = tf.image.convert_image_dtype(instance_dict['image'], dtype=tf.float32)
    image_val = tf.image.resize(image_val, (224,224), method='bilinear')
    image_val = tf.image.convert_image_dtype(image_val, dtype=tf.uint8, saturate=True)
    
    label_val = tf.image.resize(instance_dict['label'], (224,224), method='nearest') 
    
    return image_val, label_val

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # choose which GPU to run on.

    import time

    import matplotlib.pyplot as plt

    MODE = 'carla' # 'cityscapes' or 'carla'
    
    np.random.seed(0)
    if MODE == 'carla':
        TRAIN_DIR = './train/'
        VAL_DIR = './val/'

        train_imgs = np.array(glob.glob(TRAIN_DIR + 'images/*.png'))
        val_imgs = np.array(glob.glob(VAL_DIR + 'images/*.png'))

        parse_function = partial(parse_image, num_seg_classes=13, crop_bbox=[0, 0, 448, 800])
    elif MODE == 'cityscapes':
        TRAIN_DIR = '/shared/govvijay/cityscapes/leftImg8bit/train/'
        VAL_DIR   = '/shared/govvijay/cityscapes/leftImg8bit/val/'
        
        train_imgs = np.array(glob.glob(TRAIN_DIR + '*/*.png'))
        val_imgs   = np.array(glob.glob(VAL_DIR + '*/*.png'))

        parse_function = parse_image_cityscapes

    train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
    train_dataset = train_dataset.shuffle(5, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.map(tf_val_function)
    train_dataset = train_dataset.batch(8)
    #train_dataset = train_dataset.prefetch(buffer_size=32)

    run_viz = True

    start_time = time.time()
    num_batches = 0
    for ind, (img_aug, seg_aug) in enumerate(train_dataset):
        
        if run_viz:
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

            if ind > 1:
                break

        num_batches = ind

    end_time = time.time()

    print('%d Batches took %.3f seconds' % (num_batches, end_time - start_time))

###### IMGAUG CODE FOR REFERENCE #######
# This is nice but involves OpenCV under the hood.  So more CPU Usage + slower than tf.image.
# Keeping this in case I want to revert back to these more aggressive augmentations.
# def augment_function(image, label):
#     """ Takes in an image, segmentation label pair and returns augmented variants for training.
#     Note that geometric augmentations are done first to rescale/flip and get a 224x224 image.
#     Then non-geometric augmentions are done to perturb the 224x224 image for training.

#     # Reference: https://github.com/divamgupta/image-segmentation-keras/blob/bb7ac1f5cbaa884cea94e56c1a2aa5ea1690ed26/keras_segmentation/data_utils/augmentation.py#L77
#      """
#     sometimes = lambda aug: iaa.Sometimes(.75, aug)  # apply the augmentation 75% of the time
#     seq = iaa.Sequential([
#         # Step 1: Resize to (224, 224) either by direct rescaling or random scale then crop.
#         iaa.OneOf([
#             iaa.Resize(224),
#             iaa.Sequential([
#                 iaa.Resize({"height": (0.5, 0.75), "width": (0.3, 0.75)}),
#                 iaa.CropToFixedSize(width=224, height=224),
#                 ], random_order=False)
#         ]),
#         # Step 2: Flip horizontally with 50% chance.
#         iaa.Fliplr(0.5),
#         # Step 3: Apply 0 - 3 of following transforms, where each transform is applied sometimes
#         #         and in random order.
#         iaa.SomeOf((0, 3), [
#             sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
#             sometimes(iaa.LinearContrast((0.5, 1.5))),
#             sometimes(iaa.MultiplyBrightness((0.75, 1.25))),
#             sometimes(iaa.MultiplySaturation((0.5, 1.5))),
#             sometimes(iaa.Rain(speed=(0.1, 0.3))),
#             sometimes(iaa.Clouds())
#         ], random_order=True),
#         # Step 4: Randomly drop out 0 - 20% of pixels, sometimes.
#         sometimes(iaa.Dropout(p=(0, 0.2)))
#     ], random_order=False)

#     image_aug, label_aug = seq(images=image, segmentation_maps=label)  # image_aug is a list
#     return np.array(image_aug), np.array(label_aug)


# def val_augment_function(image, label):
#     """ Takes in an image, segmentation label and returns augmented variants for validation.

#     In particular, this implementation just resizes to the right size.
#     """
#     seq = iaa.Resize(224)  # 224 x 224 size for both image and label (not keeping aspect ratio)
#     image_val, label_val = seq(images=image, segmentation_maps=label) # image_val is a list
#     return np.array(image_val), np.array(label_val)

# def tf_augment_function(instance_dict):
#     """ Helper function to wrap imgaug training augmentation into a tf numpy_function """
#     image_shape = [None, 224, 224, 3]
#     label_shape = [None, 224, 224, 13]
#     [image, label] = tf.numpy_function(
#         augment_function, [instance_dict["image"], instance_dict["label"]],
#         [tf.uint8, tf.uint8])
#     image.set_shape(image_shape)
#     label.set_shape(label_shape)

#     return image, label

# def tf_val_augment_function(instance_dict):
#     """ Helper function to wrap imgaug validation augmentation into a tf numpy_function """
#     image_shape = [None, 224, 224, 3]
#     label_shape = [None, 224, 224, 13]
#     [image, label] = tf.numpy_function(
#         val_augment_function,
#         [instance_dict["image"], instance_dict["label"]],
#         [tf.uint8, tf.uint8])
#     image.set_shape(image_shape)
#     label.set_shape(label_shape)

#     return image, label