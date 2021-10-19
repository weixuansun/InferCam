import sys
sys.path.append("/home/users/u5876230/wsss/")
print(sys.path)
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from tool import imutils
from torchvision import transforms

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

classes = [{"supercategory": "person", "id": 1, "name": "person"}, # 一共80类
               {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
               {"supercategory": "vehicle", "id": 3, "name": "car"},
               {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
               {"supercategory": "vehicle", "id": 5, "name": "airplane"},
               {"supercategory": "vehicle", "id": 6, "name": "bus"},
               {"supercategory": "vehicle", "id": 7, "name": "train"},
               {"supercategory": "vehicle", "id": 8, "name": "truck"},
               {"supercategory": "vehicle", "id": 9, "name": "boat"},
               {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
               {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
               {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
               {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
               {"supercategory": "outdoor", "id": 15, "name": "bench"},
               {"supercategory": "animal", "id": 16, "name": "bird"},
               {"supercategory": "animal", "id": 17, "name": "cat"},
               {"supercategory": "animal", "id": 18, "name": "dog"},
               {"supercategory": "animal", "id": 19, "name": "horse"},
               {"supercategory": "animal", "id": 20, "name": "sheep"},
               {"supercategory": "animal", "id": 21, "name": "cow"},
               {"supercategory": "animal", "id": 22, "name": "elephant"},
               {"supercategory": "animal", "id": 23, "name": "bear"},
               {"supercategory": "animal", "id": 24, "name": "zebra"},
               {"supercategory": "animal", "id": 25, "name": "giraffe"},
               {"supercategory": "accessory", "id": 27, "name": "backpack"},
               {"supercategory": "accessory", "id": 28, "name": "umbrella"},
               {"supercategory": "accessory", "id": 31, "name": "handbag"},
               {"supercategory": "accessory", "id": 32, "name": "tie"},
               {"supercategory": "accessory", "id": 33, "name": "suitcase"},
               {"supercategory": "sports", "id": 34, "name": "frisbee"},
               {"supercategory": "sports", "id": 35, "name": "skis"},
               {"supercategory": "sports", "id": 36, "name": "snowboard"},
               {"supercategory": "sports", "id": 37, "name": "sports ball"},
               {"supercategory": "sports", "id": 38, "name": "kite"},
               {"supercategory": "sports", "id": 39, "name": "baseball bat"},
               {"supercategory": "sports", "id": 40, "name": "baseball glove"},
               {"supercategory": "sports", "id": 41, "name": "skateboard"},
               {"supercategory": "sports", "id": 42, "name": "surfboard"},
               {"supercategory": "sports", "id": 43, "name": "tennis racket"},
               {"supercategory": "kitchen", "id": 44, "name": "bottle"},
               {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
               {"supercategory": "kitchen", "id": 47, "name": "cup"},
               {"supercategory": "kitchen", "id": 48, "name": "fork"},
               {"supercategory": "kitchen", "id": 49, "name": "knife"},
               {"supercategory": "kitchen", "id": 50, "name": "spoon"},
               {"supercategory": "kitchen", "id": 51, "name": "bowl"},
               {"supercategory": "food", "id": 52, "name": "banana"},
               {"supercategory": "food", "id": 53, "name": "apple"},
               {"supercategory": "food", "id": 54, "name": "sandwich"},
               {"supercategory": "food", "id": 55, "name": "orange"},
               {"supercategory": "food", "id": 56, "name": "broccoli"},
               {"supercategory": "food", "id": 57, "name": "carrot"},
               {"supercategory": "food", "id": 58, "name": "hot dog"},
               {"supercategory": "food", "id": 59, "name": "pizza"},
               {"supercategory": "food", "id": 60, "name": "donut"},
               {"supercategory": "food", "id": 61, "name": "cake"},
               {"supercategory": "furniture", "id": 62, "name": "chair"},
               {"supercategory": "furniture", "id": 63, "name": "couch"},
               {"supercategory": "furniture", "id": 64, "name": "potted plant"},
               {"supercategory": "furniture", "id": 65, "name": "bed"},
               {"supercategory": "furniture", "id": 67, "name": "dining table"},
               {"supercategory": "furniture", "id": 70, "name": "toilet"},
               {"supercategory": "electronic", "id": 72, "name": "tv"},
               {"supercategory": "electronic", "id": 73, "name": "laptop"},
               {"supercategory": "electronic", "id": 74, "name": "mouse"},
               {"supercategory": "electronic", "id": 75, "name": "remote"},
               {"supercategory": "electronic", "id": 76, "name": "keyboard"},
               {"supercategory": "electronic", "id": 77, "name": "cell phone"},
               {"supercategory": "appliance", "id": 78, "name": "microwave"},
               {"supercategory": "appliance", "id": 79, "name": "oven"},
               {"supercategory": "appliance", "id": 80, "name": "toaster"},
               {"supercategory": "appliance", "id": 81, "name": "sink"},
               {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
               {"supercategory": "indoor", "id": 84, "name": "book"},
               {"supercategory": "indoor", "id": 85, "name": "clock"},
               {"supercategory": "indoor", "id": 86, "name": "vase"},
               {"supercategory": "indoor", "id": 87, "name": "scissors"},
               {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
               {"supercategory": "indoor", "id": 89, "name": "hair drier"},
               {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]

cls_dict = {}
for index, item in enumerate(classes):
    category_id = item['id']
    cls_dict[category_id] = index
# print(cls_dict)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    cls_labels_dict = np.load('voc12/cls_labels.npy').item()
    np.load = np_load_old
    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    #img_name_list = img_gt_name_list
    return img_name_list


class cocoImageDataset(Dataset):
    def __init__(self, voc12_root, transform=None):
        self.img_name_list = os.listdir(voc12_root)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(os.path.join(self.voc12_root, name)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return name, img


class cocoClsDataset(cocoImageDataset):

    def __init__(self, voc12_root, transform=None):
        super().__init__(voc12_root, transform)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label_file_name = name.split('.')[0]
        label_txt = open('/home/users/u5876230/coco/annotations/bbx/' + label_file_name + '.txt')
        label = label_txt.readlines()[1:]
        print(label)
        label_list = []

        multi_cls_lab = np.zeros((80), np.float32)

        for item in label:
            print(item)
            category_index = int(item.split(',')[0].split(':')[1])
            class_index = cls_dict[category_index]
            multi_cls_lab[class_index] = 1

        multi_cls_lab = torch.from_numpy(multi_cls_lab)
        label_txt.close()

        return name, img, multi_cls_lab

class cocoDatasetMSF(cocoClsDataset):
    def __init__(self, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img

class VOC12ImageDatasetAug(Dataset):
    def __init__(self, img_name_list_path, voc12_root,aug_path, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.aug_img_dir = aug_path

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        img1 = PIL.Image.open(os.path.join(self.aug_img_dir, '{}_1.jpg'.format(name))).convert("RGB")
        img2 = PIL.Image.open(os.path.join(self.aug_img_dir, '{}_2.jpg'.format(name))).convert("RGB")
        img3 = PIL.Image.open(os.path.join(self.aug_img_dir, '{}_3.jpg'.format(name))).convert("RGB")
        img4 = PIL.Image.open(os.path.join(self.aug_img_dir, '{}_4.jpg'.format(name))).convert("RGB")

        if self.transform:
            img = self.transform(img)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        return name, img, img1, img2, img3, img4


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label

class VOC12ClsDatasetAug(VOC12ImageDatasetAug):

    def __init__(self, img_name_list_path, voc12_root,aug_path, transform=None):
        super().__init__(img_name_list_path, voc12_root,aug_path, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img, img1, img2, img3, img4 = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, img1, img2, img3, img4, label

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOCImageDataset2(Dataset):
    def __init__(self, img_name_list_path, voc12_root, split_index, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform
        self.aug_img_dir = '/home/users/u5876230/fbwss_output/baseline_trainaug_aug/'
        self.split_index = split_index

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(os.path.join(self.aug_img_dir, '{}_{}.jpg'.format(name, self.split_index))).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return name, img


class VOCClsDataset2(VOCImageDataset2):
    def __init__(self,img_name_list_path, voc12_root, split_index, transform=None):
        super().__init__(img_name_list_path, voc12_root, split_index, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        multi_cls_label = torch.from_numpy(self.label_list[idx])

        return name, img, multi_cls_label


class VOCDatasetMSF2(VOCClsDataset2):
    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1, split_index=1):
        super().__init__(img_name_list_path, voc12_root, split_index, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetMSFsplit(VOC12ClsDatasetAug):
    def __init__(self, img_name_list_path, voc12_root, aug_path, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root,aug_path, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, img1, img2, img3, img4, label = super().__getitem__(idx)

        img_list = [img, img1, img2, img3, img4]
        output_list = []

        for img in img_list:
            rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

            ms_img_list = []
            for s in self.scales:
                target_size = (round(rounded_size[0]*s),
                               round(rounded_size[1]*s))
                s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
                ms_img_list.append(s_img)

            if self.inter_transform:
                for i in range(len(ms_img_list)):
                    ms_img_list[i] = self.inter_transform(ms_img_list[i])

            msf_img_list = []
            for i in range(len(ms_img_list)):
                msf_img_list.append(ms_img_list[i])
                msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            output_list.append(msf_img_list)

        return name, output_list, label

class VOC12ClsDatasetMSFsplitnew(VOC12ClsDatasetAug):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, img1, img2, img3, img4, label = super().__getitem__(idx)

        img_list = [img1, img2, img3, img4]
        msf_img_list = []

        for img in img_list:
            rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

            ms_img_list = []
            for s in self.scales:
                target_size = (round(rounded_size[0]*s),
                               round(rounded_size[1]*s))
                s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
                ms_img_list.append(s_img)

            if self.inter_transform:
                for i in range(len(ms_img_list)):
                    ms_img_list[i] = self.inter_transform(ms_img_list[i])

            for i in range(len(ms_img_list)):
                msf_img_list.append(ms_img_list[i])
                msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label

class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        label_la = np.load(label_la_path).item()
        label_ha = np.load(label_ha_path).item()
        np.load = np_load_old

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0))

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label

if __name__ == '__main__':
    import importlib
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_gcn", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="gcn", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root", default = '/home/users/u5876230/pascal_aug/VOCdevkit/VOC2012/', type=str)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    train_dataset = VOC12ClsDatasetAug(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(256, 512),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    class_clount_dict = {}

    for iter, pack in enumerate(train_data_loader):
        print(len(pack))
        img = pack[3]
        label = pack[6]
        # print(img.shape)
        # class_count = (int(torch.sum(label)))
        # if class_count in class_clount_dict:
        #     class_clount_dict[class_count] += 1
        # else:
        #     class_clount_dict[class_count] = 1
        #
        # print(class_clount_dict)