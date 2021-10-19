import numpy as np
import torch
import math
from scipy import ndimage
import os
import pickle
import PIL.Image
import cv2
import random

dataset_path = '/home/users/u5876230/wsss/voc12/train_aug.txt'
jpg_img_path = '/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/JPEGImages'
mass_center_dict_path = '/home/users/u5876230/fbwss_output/baseline_trainaug_masscenter'
img_gt_name_list = open(dataset_path).read().splitlines()
img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

# img_name_list = os.listdir(mass_center_dict_path)

for index, name in enumerate(img_name_list):
    img = cv2.imread(os.path.join(jpg_img_path, '{}.jpg'.format(name)))
    h, w, c = img.shape
    print(index)
    dict_file = open(os.path.join(mass_center_dict_path, '{}.pkl'.format(name)), 'rb')
    mass_center_dict = pickle.load(dict_file)
    dict_file.close()
    mass_center_mat = np.array(list(mass_center_dict.values()))

    # keys = list(mass_center_dict.keys())
    # for target_class in keys:
    #     mass_center = mass_center_dict[target_class]
        # cv2.circle(img, (mass_center[1], mass_center[0]), 4, (0, 0, 255), 4)
    # cv2.imwrite('tmp/{}.jpg'.format(name), img)

    if 1 < len(mass_center_dict) < 3:
        buffer_size = 20
        img_1 = img[0:np.max(mass_center_mat[:, 0]) + buffer_size, 0:np.max(mass_center_mat[:, 1]) + buffer_size]
        img_2 = img[0:np.max(mass_center_mat[:, 0]) + buffer_size, np.min(mass_center_mat[:, 1]) - buffer_size:w]
        img_3 = img[np.min(mass_center_mat[:, 0]) - buffer_size:h, 0:np.max(mass_center_mat[:, 1]) + buffer_size]
        img_4 = img[np.min(mass_center_mat[:, 0]) - buffer_size:h, np.min(mass_center_mat[:, 1]) - buffer_size:w]
    elif len(mass_center_dict) == 1:
        buffer_size = 10
        img_1 = img[0:np.max(mass_center_mat[:, 0]) + buffer_size, 0:np.max(mass_center_mat[:, 1]) + buffer_size]
        img_2 = img[0:np.max(mass_center_mat[:, 0]) + buffer_size, np.min(mass_center_mat[:, 1]) - buffer_size:w]
        img_3 = img[np.min(mass_center_mat[:, 0]) - buffer_size:h, 0:np.max(mass_center_mat[:, 1]) + buffer_size]
        img_4 = img[np.min(mass_center_mat[:, 0]) - buffer_size:h, np.min(mass_center_mat[:, 1]) - buffer_size:w]
    elif len(mass_center_dict) > 2:
        img_1 = img_2 = img_3 = img_4 = img

    cv2.imwrite('/home/users/u5876230/fbwss_output/baseline_trainaug_aug/{}_1.jpg'.format(name), img_1)
    cv2.imwrite('/home/users/u5876230/fbwss_output/baseline_trainaug_aug/{}_2.jpg'.format(name), img_2)
    cv2.imwrite('/home/users/u5876230/fbwss_output/baseline_trainaug_aug/{}_3.jpg'.format(name), img_3)
    cv2.imwrite('/home/users/u5876230/fbwss_output/baseline_trainaug_aug/{}_4.jpg'.format(name), img_4)








