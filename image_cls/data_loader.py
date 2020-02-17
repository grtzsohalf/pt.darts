from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils


class ImageDataset(Dataset):
    def _load_data(self, npy_file):
        data = np.load(npy_file)
        print('Shape: ', data.shape)
        return data

    def __init__(self, root, split='train+val', transform=None, target_transform=None):

        self.root = root

        if split == 'train+val':
            img_npy = os.path.join(self.root, 'Train_img.npy')
            lbl_npy = os.path.join(self.root, 'train_label.npy')
            train_data = self._load_data(img_npy)
            train_targets = self._load_data(lbl_npy)
            img_npy = os.path.join(self.root, 'Val_img.npy')
            lbl_npy = os.path.join(self.root, 'val_label.npy')
            self.data = np.concatenate((train_data, self._load_data(img_npy)), axis=0)
            self.targets = np.concatenate((train_targets, self._load_data(lbl_npy)), axis=0)
        elif split == 'train':
            img_npy = os.path.join(self.root, 'Train_img.npy')
            lbl_npy = os.path.join(self.root, 'train_label.npy')
            self.data = self._load_data(img_npy)
            self.targets = self._load_data(lbl_npy)
        elif split == 'val':
            img_npy = os.path.join(self.root, 'Val_img.npy')
            lbl_npy = os.path.join(self.root, 'val_label.npy')
            self.data = self._load_data(img_npy)
            self.targets = self._load_data(lbl_npy)
        elif split == 'test':
            img_npy = os.path.join(self.root, 'Test_img.npy')
            lbl_npy = os.path.join(self.root, 'test_label.npy')
            self.data = self._load_data(img_npy)
            self.targets = self._load_data(lbl_npy)
        
        # self.img_mean = [np.mean(self.data/255)]
        # self.img_std = [np.std(self.data/255)]
        # print("Mean: ", self.img_mean)
        # print("Std: ", self.img_std)
        # self.train_transform = self._train_data_transform()
        # self.test_transform = self._test_data_transform()

        self.transform = transform
        self.target_transform = target_transform

        self.data = self.data.transpose((0, 2, 3, 1)) # transpose to HWC

        assert len(self.data) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(np.uint8(np.squeeze(img)))

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        # target = self.target_transform(target)

        return img, target

    def add_data(self, dataset):
        self.data = np.concatenate((self.data, dataset.data), axis=0)
        self.targets = np.concatenate((self.targets, dataset.targets), axis=0)

    def _train_data_transform(self):
        data_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ])
        # if args.cutout:
        # data_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))
        return data_transform

    def _test_data_transform(self):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ])
        # if args.cutout:
        # data_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))
        return data_transform


if __name__ == '__main__':
    base_path = '/home/grtzsohalf/Desktop/NVIDIA/image_data'
    train_img_npy = os.path.join(base_path, 'Train_img.npy')
    train_lbl_npy = os.path.join(base_path, 'train_label.npy')
    test_img_npy = os.path.join(base_path, 'Test_img.npy')
    test_lbl_npy = os.path.join(base_path, 'test_label.npy')
    val_img_npy = os.path.join(base_path, 'Val_img.npy')
    val_lbl_npy = os.path.join(base_path, 'val_label.npy')

    train_img_dataset = ImageDataset(train_img_npy, train_lbl_npy)
    # test_img_dataset = ImageDataset(test_img_npy, test_lbl_npy, train_img_dataset.img_mean, train_img_dataset.img_std)
    # val_img_dataset = ImageDataset(val_img_npy, val_lbl_npy, train_img_dataset.img_mean, train_img_dataset.img_std)

    fig = plt.figure()

    print(train_img_dataset.data[0][0])
    sample = train_img_dataset[200][0].reshape(96,160)
    imgplot = plt.imshow(sample)
    plt.show()
