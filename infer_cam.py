
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os
import cv2
from scipy import ndimage
import pickle
from datetime import datetime
import time


classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa', 'train','tvmonitor']


def infer_split_cam(args):
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    infer_dataset = voc12.data.VOC12ClsDatasetMSFsplit(args.infer_list, voc12_root=args.voc12_root, 
                                                        aug_path = args.aug_path, scales=(1, 0.5, 1.5, 2.0),
                                                        inter_transform=torchvision.transforms.Compose(
                                                    [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # cam_mask_dict = {}
    for iter, (img_name, output_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]
        cam_dict = {}
        for split_index, img_list in enumerate(output_list):
            if split_index == 0:   # original image
                orig_img = cv2.imread(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name))
                raw_img = orig_img
                # raw_img = np.asarray(Image.open(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name)))
                cam_mask = np.zeros((orig_img.shape[0], orig_img.shape[1], 4, 20))
                orig_img_size = orig_img.shape[:2]
                cam_matrix = np.zeros((20, orig_img_size[0], orig_img_size[1]))
                last_left_area_matrix = np.ones((orig_img_size[0], orig_img_size[1]))
                pixel_sum = orig_img_size[0] * orig_img_size[1]
            else: # each split
                aug_img_dir = args.aug_path
                orig_img = cv2.imread(os.path.join(aug_img_dir, '{}_{}.jpg'.format(img_name, split_index)))
                orig_img_size = orig_img.shape[:2]
                cam_matrix = np.zeros((20, orig_img_size[0], orig_img_size[1]))
                last_left_area_matrix = np.ones((orig_img_size[0], orig_img_size[1]))
                pixel_sum = orig_img_size[0] * orig_img_size[1]
                orig_img_size = orig_img.shape[:2]

                split_start = datetime.now()

                for dropout_index in range(5):
                    start = datetime.now()
                    def _work(i, img):
                        with torch.no_grad():
                            with torch.cuda.device(i%n_gpus):
                                cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                                cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                                if i % 2 == 1:
                                    cam = np.flip(cam, axis=-1)
                                return cam

                    if dropout_index > 0:
                        new_img_list = []
                        for img_index, img in enumerate(img_list):
                            _, _, h, w = img.shape
                            left_area_matrix = F.interpolate(left_area_matrix, (h, w))
                            rgb_mean = torch.mean(img, dim=(2,3))
                            mask_mean = torch.ones_like(img)
                            mask_mean[:, 0, :, :] = mask_mean[:,0,:,:] * rgb_mean[0,0]
                            mask_mean[:, 1, :, :] = mask_mean[:,1,:,:] * rgb_mean[0,1]
                            mask_mean[:, 2, :, :] = mask_mean[:,2,:,:] * rgb_mean[0,2]
                            # 
                            # mask_mean[:, 0, :, :] = mask_mean[:, 0, :, :] * (122.675 / 255)
                            # mask_mean[:, 1, :, :] = mask_mean[:, 1, :, :] * (116.669 / 255)
                            # mask_mean[:, 2, :, :] = mask_mean[:, 2, :, :] * (104.008 / 255)

                            new_img = ((left_area_matrix * img) + (1-left_area_matrix) * mask_mean).float()
                            new_img_list.append(new_img)
                    else:
                        new_img_list = img_list

                    thread_pool = pyutils.BatchThreader(_work, list(enumerate(new_img_list)),
                                                        batch_size=12, prefetch_size=0, processes=args.num_workers)

                    cam_list = thread_pool.pop_results()

                    sum_cam = np.sum(cam_list, axis=0)
                    norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

                    cam_matrix = np.stack((cam_matrix, norm_cam), axis=3)
                    cam_matrix = np.max(cam_matrix, axis=3)

                    # left_area_matrix = 1 - np.max(cam_matrix, axis=0)
                    left_area_matrix = ((np.max(cam_matrix, axis=0) < 0.7) * 1)
                    cam_mask_diff = (np.sum(last_left_area_matrix) - np.sum(left_area_matrix)) / pixel_sum
                    last_left_area_matrix = left_area_matrix
                    left_area_matrix = torch.from_numpy(left_area_matrix).unsqueeze(0).unsqueeze(0).float()
                    if cam_mask_diff < 0.01:
                        break

            if split_index > 0:
                for i in range(20):
                    if label[i] > 1e-5:
                        if split_index == 1:
                            cam_mask[0:orig_img_size[0], 0:orig_img_size[1], 0, i] = cam_matrix[i]
                        elif split_index == 2:
                            cam_mask[0:orig_img_size[0], cam_mask.shape[1]-orig_img_size[1]:cam_mask.shape[1], 1, i] = cam_matrix[i]
                        elif split_index == 3:
                            cam_mask[cam_mask.shape[0]-orig_img_size[0]:cam_mask.shape[0], 0:orig_img_size[1], 2, i] = cam_matrix[i]
                        elif split_index == 4:
                            cam_mask[cam_mask.shape[0]-orig_img_size[0]:cam_mask.shape[0], cam_mask.shape[1]-orig_img_size[1]:cam_mask.shape[1], 3, i] = cam_matrix[i]
                            cam_dict[i] = np.max(cam_mask[:,:,:,i], 2)

            if split_index == 4:
                if args.heatmap is not None:
                    img = raw_img
                    keys = list(cam_dict.keys())
                    for target_class in keys:
                        mask = cam_dict[target_class]
                        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                        img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0] ))
                        cam_output = heatmap * 0.3 + img * 0.5

                        cv2.imwrite(os.path.join(args.heatmap, img_name + '_{}.jpg'.format(classes[target_class])), cam_output)

                raw_img = np.asarray(Image.open(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name)))

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(cam_matrix[0])*0.2]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
        print(iter)


if __name__ == '__main__':
    total_start = datetime.now()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--split_path", default='/home/users/u5876230/fbwss_output/baseline_trainaug_aug/', type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--heatmap", default=None, type=str)
    parser.add_argument("--out_cam_pred", default=None, type=str)

    args = parser.parse_args()
    infer_split_cam(args)

