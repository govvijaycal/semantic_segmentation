import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

CARLA_LABEL_COLORS = [
    (0, 0, 0),         # None
    (70, 70, 70),      # Buildings
    (190, 153, 153),   # Fences
    (72, 0, 90),       # Other
    (220, 20, 60),     # Pedestrians
    (153, 153, 153),   # Poles
    (157, 234, 50),    # RoadLines
    (128, 64, 128),    # Roads
    (244, 35, 232),    # Sidewalks
    (107, 142, 35),    # Vegetation
    (0, 0, 255),      # Vehicles
    (102, 102, 156),  # Walls
    (220, 220, 0)]     # TrafficSigns


def convert_label_to_palette_img(img_label):
    h, w = img_label.shape

    palette_img = np.zeros((h,w,3)).astype(np.uint8)

    for c in range(len(CARLA_LABEL_COLORS)):
        palette_img[:,:,0] += ((img_label == c) * CARLA_LABEL_COLORS[c][0]).astype('uint8')
        palette_img[:,:,1] += ((img_label == c) * CARLA_LABEL_COLORS[c][1]).astype('uint8')
        palette_img[:,:,2] += ((img_label == c) * CARLA_LABEL_COLORS[c][2]).astype('uint8')

    return palette_img


if __name__ == '__main__':
    srcdir  = '../example_50_ex/seg_50_ex/'
    destdir = '../example_50_ex/pal_50_ex/'

    imgs = glob.glob(srcdir + '*.png')

    for img in imgs:
        im_pil = np.array(Image.open(img))
        im_pal = convert_label_to_palette_img(im_pil)
        print('saving to: ' +  destdir + img.split('/')[-1])
        Image.fromarray(im_pal).save(destdir + img.split('/')[-1])
