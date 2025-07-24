#from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter

from skimage.draw import polygon as draw_polygon
from scipy.spatial import ConvexHull



def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask

def random_rotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask



def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def vflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def rotate(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.ROTATE_90)
        mask = mask.transpose(Image.ROTATE_90)
    return img, mask


class KVASIRDataset_H5(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        
        id = self.ids[item]+'.h5'
        if 'train' in self.mode:
            h5f = h5py.File(self.root + "/train/{}".format(id), 'r')
        
        if self.mode == 'val':
            h5f = h5py.File(self.root + "/val/{}".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            return img, mask

        img, mask = h5f['image'][:], h5f['label'][:]
        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype(np.uint8))
        mask = Image.fromarray(mask * 255)

        if random.random() > 0.5:
            img, mask = hflip(img, mask)
        if random.random() > 0.5:
            img, mask = vflip(img, mask)
        if random.random() > 0.5:
            img, mask = rotate(img, mask)
       
        if self.mode == 'train_l':
            img, mask = np.asarray(img) / 255.0, np.asarray(mask) // 255
            img = img.transpose((2, 0, 1))
            return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask)).long()

        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1).transpose((2, 0, 1))).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2).transpose((2, 0, 1))).float() / 255.0

        return img, img_s1, img_s2, cutmix_box1, cutmix_box2
        
    def __len__(self):
        return len(self.ids)
    
class KVASIRFS_H5(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(self.root+'/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        
        id = self.ids[item]+'.h5'
        if 'train' in self.mode:
            h5f = h5py.File(self.root + "/train/{}".format(id), 'r')
        
        if self.mode == 'val':
            h5f = h5py.File(self.root + "/val/{}".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            return img, mask

        img, mask = h5f['image'][:], h5f['label'][:]
        img = Image.fromarray((img.transpose((1, 2, 0)) * 255).astype(np.uint8))
        mask = Image.fromarray(mask * 255)

        if random.random() > 0.5:
            img, mask = hflip(img, mask)
        if random.random() > 0.5:
            img, mask = vflip(img, mask)
        if random.random() > 0.5:
            img, mask = rotate(img, mask)
       
        #if self.mode == 'train_l':
        #    img, mask = np.asarray(img) / 255.0, np.asarray(mask) // 255
        #    img = img.transpose((2, 0, 1))
        #    return torch.from_numpy(img).float(), torch.from_numpy(np.array(mask)).long()

        img_s1 = deepcopy(img)
        img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
        mask = np.asarray(mask) // 255
        mask = torch.from_numpy(np.array(mask)).long()
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1).transpose((2, 0, 1))).float() / 255.0

        return img, img_s1, mask, cutmix_box1
    def __len__(self):
        return len(self.ids)