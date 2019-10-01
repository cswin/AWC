#The code is modified from https://github.com/HzFu/MNet_DeepCDR

#Please go to https://refuge.grand-challenge.org/ to download the data

# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

import Model_DiscSeg as DiscModel
from mnet_utils import BW_img, disc_crop, mk_dir, files_with_ext

# disc_list = [400, 500, 600, 700, 800] #please note: during training, the size of images is same, which is only one in the list not all of them.
disc_list = [600] # 460x460 for validation and test dataset;  600x600 for training
DiscROI_size = 800  # cropped image size  and will be updated based on disc_list
DiscSeg_size = 640  # input size to the disc detection model


data_type = '.jpg'
data_img_path = path.join('../../data', 'Training400')
label_img_path = path.join('../../data', 'Annotation-Training400', 'Disc_Cup_Masks')

data_save_path = mk_dir(path.join('../../data', 'train_crop_s600', 'data'))
label_save_path = mk_dir(path.join('../../data', 'train_crop_s600', 'label'))

file_test_list = files_with_ext(data_img_path, data_type)

DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('Model_DiscSeg_ORIGA.h5')

Disc_flat = None

is_polar_transform = False

for lineIdx, temp_txt in enumerate(file_test_list):
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))

    # load image
    org_img = np.asarray(image.load_img(path.join(data_img_path, temp_txt)))

    # load label
    org_label = np.asarray(image.load_img(path.join(label_img_path, temp_txt[:-4] + '.bmp')))[:, :, 0]


    new_label = np.zeros(np.shape(org_label) + (3,), dtype=np.uint8)
    new_label[org_label < 200, 0] = 255
    new_label[org_label < 100, 1] = 255

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])

    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)

    for disc_idx, DiscROI_size in enumerate(disc_list):


        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)

        label_region, _, _ = disc_crop(new_label, DiscROI_size, C_x, C_y)

        if is_polar_transform:
            Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                               cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)

            Label_flat = rotate(cv2.linearPolar(label_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                            cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)


        if is_polar_transform:
            disc_result = Image.fromarray((Disc_flat * 255).astype(np.uint8))
        else:
            disc_result = Image.fromarray(disc_region)
        filename = '{}_{}.bmp'.format(temp_txt[:-4], DiscROI_size)
        disc_result.save(path.join(data_save_path, filename))

        if is_polar_transform:
            Label_flat = (Label_flat * 255).astype(np.uint8)
            Label_flat[Label_flat > 200] = 255
        else:
            Label_flat = label_region

        label_result = Image.fromarray(Label_flat)
        label_result.save(path.join(label_save_path, filename))

if is_polar_transform:
    plt.imshow(Disc_flat)
else:
    plt.imshow(disc_region)
plt.show()
