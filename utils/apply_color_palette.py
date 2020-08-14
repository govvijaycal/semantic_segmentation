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
    (0, 0, 255),       # Vehicles
    (102, 102, 156),   # Walls
    (220, 220, 0)]     # TrafficSigns

CITY_LABEL_COLORS = [
    (128, 64,128), # Road
    (244, 35,232), # Sidewalk
    ( 70, 70, 70), # Building
    (102,102,156), # Wall
    (190,153,153), # Fence
    (153,153,153), # Pole
    (250,170, 30), # Traffic Light
    (220,220,  0), # Traffic Sign
    (107,142, 35), # Vegetation
    (152,251,152), # Terrain
    ( 70,130,180), # Sky
    (220, 20, 60), # Person (pedestrian)
    (255,  0,  0), # Rider (cyclist, etc.)
    (  0,  0,142), # Car
    (  0,  0, 70), # Truck
    (  0, 60,100), # Bus
    (  0, 80,100), # Train
    (  0,  0,230), # Motorcycle
    (119, 11, 32), # Bicycle
    (  0,  0,  0), # Background/Ignore
]


def convert_label_to_palette_img(img_label, palette):
    h, w = img_label.shape

    palette_img = np.zeros((h,w,3)).astype(np.uint8)

    for c in range(len(palette)):
        palette_img[:,:,0] += ((img_label == c) * palette[c][0]).astype('uint8')
        palette_img[:,:,1] += ((img_label == c) * palette[c][1]).astype('uint8')
        palette_img[:,:,2] += ((img_label == c) * palette[c][2]).astype('uint8')

    return palette_img


if __name__ == '__main__':
    srcdir  = '../example_50_ex/seg_50_ex/'
    destdir = '../example_50_ex/pal_50_ex/'

    imgs = glob.glob(srcdir + '*.png')

    for img in imgs:
        im_pil = np.array(Image.open(img))
        im_pal = convert_label_to_palette_img(im_pil, CARLA_LABEL_COLORS)
        print('saving to: ' +  destdir + img.split('/')[-1])
        Image.fromarray(im_pal).save(destdir + img.split('/')[-1])
