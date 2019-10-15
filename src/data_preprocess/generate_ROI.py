import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import numpy as np
import scipy.io as sio
import scipy.misc
from keras.preprocessing import image
from skimage.transform import rotate, resize
from skimage.measure import label, regionprops
from time import time
from mnet_utils import pro_process, BW_img, disc_crop
import matplotlib.pyplot as plt
from skimage.io import imsave

import Model_DiscSeg as DiscModel

ROI_size_list = [400, 500, 600, 700, 800]
DiscROI_size = 600 # 460x460 for validation and test dataset;  600x600 for training
DiscSeg_size = 640  # input size to the disc detection model


train_data_type = '.jpg'
mask_data_type = '.bmp'

Original_vali_img_path = '../../data/Training400/'
Original_Mask_img_path = '../../data/Annotation-Training400/Disc_Cup_Masks/'

Image_save_path = '../../data/train_crop_polar_sall/data/'
MaskImage_save_path = '../../data/train_crop_polar_sall/label/'

if not os.path.exists(Image_save_path):
    os.makedirs(Image_save_path)

if not os.path.exists(MaskImage_save_path):
    os.makedirs(MaskImage_save_path)

is_polar_coordinate =True  # in MICCAI version, this is false.


file_train_list = [file for file in os.listdir(Original_vali_img_path) if file.lower().endswith(train_data_type)]
print(str(len(file_train_list)))


DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('Model_DiscSeg_ORIGA.h5')


for lineIdx in range(0, len(file_train_list)):


    ##########################Generate mask ROIs #############################


    temp_txt = [elt.strip() for elt in file_train_list[lineIdx].split(',')]
    # print(' Processing Img: ' + temp_txt[0])
    # load image
    org_img = np.asarray(image.load_img(Original_vali_img_path + temp_txt[0]))
    # plt.imshow(org_img)
    # plt.title('org_img')
    # plt.show()

    nameLen = len(temp_txt[0])
    org_mask = np.asarray(image.load_img(Original_Mask_img_path +
                                         temp_txt[0][:nameLen - 4] + mask_data_type))[:, :, 0]
    # plt.imshow(org_mask)
    # plt.title('org_mask')
    # plt.show()

    org_disc = org_mask < 255
    # plt.imshow(org_disc)
    # plt.title('org_disc')
    # plt.show()

    org_cup = org_mask == 0
    # plt.imshow(org_cup)
    # plt.title('org_cup')
    # plt.show()

    # Disc region detection by U-Net
    temp_org_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3))
    # plt.imshow(temp_org_img)
    # plt.title('temp_org_img')
    # plt.show()

    temp_org_mask = resize(org_mask, (DiscSeg_size, DiscSeg_size))
    # plt.imshow(temp_org_mask)
    # plt.title('temp_org_mask')
    # plt.show()

    temp_org_disc = resize(org_disc, (DiscSeg_size, DiscSeg_size))
    # plt.imshow(temp_org_disc)
    # plt.title('temp_org_disc')
    # plt.show()

    temp_org_cup = resize(org_cup, (DiscSeg_size, DiscSeg_size))
    # plt.imshow(temp_org_cup)
    # plt.title('temp_org_cup')
    # plt.show()

    temp_org_img = np.reshape(temp_org_img, (1,) + temp_org_img.shape) * 255

    prob_10 = DiscSeg_model.predict([temp_org_img])

    # plt.imshow(np.squeeze(np.clip(prob_10*255,0,255).astype('uint8')))
    # plt.title('temp_img')
    # plt.show()

    org_img_disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
    org_disc_bw = BW_img(np.reshape(temp_org_disc, (DiscSeg_size, DiscSeg_size)), 0.5)
    org_cup_bw = BW_img(np.reshape(temp_org_cup, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(org_img_disc_map))

    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    for disc_idx, DiscROI_size in enumerate(ROI_size_list):

        org_img_disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        # plt.imshow(org_img_disc_region)
        # plt.title('org_img_disc_region')
        # plt.show()

        org_mask_region, err_coord, crop_coord = disc_crop(org_mask, DiscROI_size, C_x, C_y)
        org_disc_region, err_coord_disc, crop_coord_disc = disc_crop(org_disc, DiscROI_size, C_x, C_y)
        # plt.imshow(org_disc_region)
        # plt.title('org_disc_region')
        # plt.show()

        org_cup_region, err_coord_cup, crop_coord_cup = disc_crop(org_cup, DiscROI_size, C_x, C_y)
        # plt.imshow(org_cup_region)
        # plt.title('org_cup_region')
        # plt.show()

        ROI_mask_result = np.array(BW_img(org_disc_region, 0.5), dtype=int) + np.array(BW_img(org_cup_region, 0.5),
                                                                                       dtype=int)
        # plt.imshow(ROI_mask_result)
        # plt.title('ROI_mask_result')
        # plt.show()

        ROI_mask_result = (255.0 / ROI_mask_result.max() * (ROI_mask_result - ROI_mask_result.min())).astype(np.uint8)
        ROI_mask_result[ROI_mask_result == 255] = 200
        ROI_mask_result[ROI_mask_result == 0] = 255
        ROI_mask_result[ROI_mask_result == 200] = 0
        ROI_mask_result[(ROI_mask_result > 0) & (ROI_mask_result < 255)] = 128
        # plt.imshow(ROI_mask_result)
        # plt.title('ROI_mask_result')
        # plt.show()

        if is_polar_coordinate:
            ROI_mask_result = rotate(cv2.linearPolar(ROI_mask_result, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                            cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)

            ROI_img_result = rotate(cv2.linearPolar(org_img_disc_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                            cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)

        # plt.imshow(ROI_mask_result)
        # plt.title('ROI_mask_result')
        # plt.show()

        filename_ROI_Img = '{}_{}{}'.format(temp_txt[0][:nameLen - 4], DiscROI_size, train_data_type)
        filename_ROI_Mask = '{}_{}{}'.format(temp_txt[0][:nameLen - 4], DiscROI_size, mask_data_type)
        imsave(Image_save_path + filename_ROI_Img, ROI_img_result)
        imsave(MaskImage_save_path + filename_ROI_Mask, ROI_mask_result)

    ##########################Generate mask ROI done #############################


