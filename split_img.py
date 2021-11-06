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
import os.path
import os
import cv2
from scipy import ndimage
import pickle
from datetime import datetime


classes = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']

if __name__ == '__main__':
    start = datetime.now()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--infer_list", default="voc12/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--voc12_root", default='/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--heatmap", default=None, type=str)
    parser.add_argument("--split_path", default=None, type=str)

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=(1, 0.5, 1.5, 2.0),
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    print('Start splitting images:')

    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]
        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        mass_center_dict = {} # get mass center of every cam mask

        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]
                try:
                    current_mass_center = list(map(int, ndimage.measurements.center_of_mass(norm_cam[i])))
                except:
                    h, w, c = orig_img.shape
                    current_mass_center = [int(h/2), int(w/2)]
                mass_center_dict[i] = current_mass_center
        
        img = cv2.imread(args.voc12_root + '/JPEGImages/{}.jpg'.format(img_name))
        h, w, c = img.shape
        mass_center_mat = np.array(list(mass_center_dict.values()))

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
            img_1 = img_2 = img_3 = img_4 = img # if more than 2 classes, don't split 

        cv2.imwrite(os.path.join(args.split_path, '{}_1.jpg'.format(img_name)), img_1)
        cv2.imwrite(os.path.join(args.split_path, '{}_2.jpg'.format(img_name)), img_2)
        cv2.imwrite(os.path.join(args.split_path, '{}_3.jpg'.format(img_name)), img_3)
        cv2.imwrite(os.path.join(args.split_path, '{}_4.jpg'.format(img_name)), img_4)

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        if args.heatmap is not None:
            # aug_img_dir = '/home/users/u5876230/fbwss_output/baseline_trainaug_aug/'
            # img = cv2.imread(os.path.join(aug_img_dir, '{}_3.jpg'.format(img_name)))

            keys = list(cam_dict.keys())
            for target_class in keys:
                mask = cam_dict[target_class]
                heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
                img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0] ))
                cam_output = heatmap * 0.3 + img * 0.5
                cv2.imwrite(os.path.join(args.heatmap, img_name + '_{}.jpg'.format(classes[target_class])), cam_output)

      
        print(iter)


        print(datetime.now() - start)









