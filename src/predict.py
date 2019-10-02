# -*- coding: utf-8 -*-
# this code is modified from https://github.com/HzFu/MNet_DeepCDR/tree/master/mnet_deep_cdr

from __future__ import print_function
from os import path
from time import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

from data_preprocess.Model_DiscSeg import DeepModel
from data_preprocess.mnet_utils import pro_process, BW_img, disc_crop, mk_dir, files_with_ext
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from models.unet import UNet

DiscROI_size = 600
DiscSeg_size = 640
CDRSeg_size = 400

parent_dir = "../data/"
gpu = 1

test_data_path = path.join(parent_dir, 'Validation400')
data_save_path_ROI = mk_dir(path.join(parent_dir, 'resultROI'))
data_save_path_whole = mk_dir(path.join(parent_dir, 'resultWhole'))

file_test_list = files_with_ext(test_data_path, '.jpg')

DiscSeg_model = DeepModel(size_set=DiscSeg_size)
DiscSeg_model.load_weights('./data_preprocess/Model_DiscSeg_ORIGA.h5')

model_path = path.join(parent_dir, 'snapshots', "UNet80000v7.pth")

JoinSeg_model = UNet(3, n_classes=2)
saved_state_dict = torch.load(model_path)
JoinSeg_model.load_state_dict(saved_state_dict)

JoinSeg_model.eval()

for lineIdx, temp_txt in enumerate(file_test_list):
    # load image
    org_img = np.asarray(image.load_img(path.join(test_data_path, temp_txt)))
    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])
    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    disc_region, err_xy, crop_xy = disc_crop(org_img, DiscROI_size, C_x, C_y)

    # Disc and Cup segmentation by M-Net
    run_start = time()
    Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2),
                                       DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS), -90)

    temp_img = pro_process(Disc_flat, CDRSeg_size)
    plt.imshow(np.squeeze(temp_img))
    plt.show()

    temp_img = temp_img.astype(np.float64)
    temp_img = temp_img.transpose(2, 0, 1)
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    temp_img = torch.from_numpy(temp_img).float()
    temp_img = Variable(temp_img, volatile=True)

    _, _, _, prob_9, prob_10 = JoinSeg_model(temp_img)
    prob_10 = prob_10.cpu().data.numpy()
    run_end = time()

    # Extract mask
    prob_map = np.reshape(prob_10, (prob_10.shape[2], prob_10.shape[3], prob_10.shape[1]))
    disc_map = np.array(Image.fromarray(prob_map[:, :, 0]).resize((DiscROI_size, DiscROI_size)))
    plt.imshow((disc_map*255).astype('uint8'))
    plt.show()
    cup_map = np.array(Image.fromarray(prob_map[:, :, 1]).resize((DiscROI_size, DiscROI_size)))
    plt.imshow((cup_map * 255).astype('uint8'))
    plt.show()


    De_disc_map = cv2.linearPolar(rotate(disc_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                  DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

    plt.imshow(De_disc_map)
    plt.show()

    De_cup_map = cv2.linearPolar(rotate(cup_map, 90), (DiscROI_size / 2, DiscROI_size / 2),
                                 DiscROI_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    plt.imshow(De_cup_map)
    plt.show()

    De_disc_map = np.array(BW_img(De_disc_map, 0.5), dtype=int)
    plt.imshow(De_disc_map)
    plt.show()

    De_cup_map = np.array(BW_img(De_cup_map, 0.8), dtype=int)
    plt.imshow(De_cup_map)
    plt.show()

    print('Processing Img {idx}: {temp_txt}, running time: {running_time}'.format(
        idx=lineIdx + 1, temp_txt=temp_txt, running_time=run_end - run_start
    ))

    # Save raw mask
    ROI_result = np.array(BW_img(De_disc_map, 0.5), dtype=int) + np.array(BW_img(De_cup_map, 0.8), dtype=int)
    ROI_result = (ROI_result * 127).astype(np.uint8)
    plt.imshow(ROI_result)
    plt.show()
    ROI_result[ROI_result>200]=255
    ROI_result[ROI_result<10]=0
    ROI_result[ROI_result==255]=25
    ROI_result[ROI_result==0]=255
    ROI_result[ROI_result==25]=0
    plt.imshow(ROI_result)
    plt.show()

    ROI_result_save = Image.fromarray(ROI_result)
    ROI_result_save.save(path.join(data_save_path_ROI, temp_txt[:-4] + '_ROI.png'))

    Img_result = np.ones((org_img.shape[0], org_img.shape[1]), dtype=np.int8)*255
    plt.imshow(Img_result)
    plt.show()
    Img_result[crop_xy[0]:crop_xy[1], crop_xy[2]:crop_xy[3], ] = ROI_result[err_xy[0]:err_xy[1], err_xy[2]:err_xy[3], ]
    plt.imshow(Img_result)
    plt.show()
    save_result = Image.fromarray(Img_result.astype(np.uint8))
    save_result.save(path.join(data_save_path_whole, temp_txt[:-4] + '.png'))
