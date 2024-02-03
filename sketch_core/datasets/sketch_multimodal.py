import os
import os.path

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from scipy.spatial.ckdtree import cKDTree as kdtree
from PIL import Image
import torch
import cv2
from torchvision import transforms as T
from torchvision.transforms import autoaugment


__all__ = ['ImageStrokeData']

class ImageStrokeData(dict):

    def __init__(self, ):

        super().__init__({
            'train':
                ImageStroke(split='train'),
            'test':
                ImageStroke(split='val')})


class ImageStroke:

    def __init__(self, split):
        self.data = []
        self.label = []
        partition = split
        self.voxel_size = 1.0
        with open(f"pointcloudsxy/tiny_{partition}_set.txt", "r") as f:
            for lines in f.readlines():
                line_list = lines.strip("\n").split(" ")
                # print(line_list)
                self.data.append(os.path.join(f"pointclouds_xy/{partition}", line_list[0]))
                self.label.append(int(line_list[1]))
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_size = [256, 256]
        if partition == "train":
            self.transform = T.Compose([
                autoaugment.TrivialAugmentWide(),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize(input_size),
                T.ToTensor(),
                normalize,
                # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(input_size),
                T.ToTensor(),
                normalize
            ])

        self.partition = partition

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.open(self.data[index].replace("pointclouds_xy", "/media/kemove/403/TEI").replace(".txt", ".png"))
        if self.transform is not None:
            img = self.transform(img)


        block_ = np.load(self.data[index].replace(".txt", ".npy")).astype(np.float32)
        # block_ = np.load(self.data[index].replace(".txt", ".npy")).astype(np.float32)
        block = np.zeros_like(block_)

        if 'train' in self.partition:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = 0
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        label = torch.tensor(self.label[index], dtype=torch.long)

        feat_ = block
        feat_center = np.mean(feat_[:, :3], axis=0, keepdims=True)
        feat_center_offset = feat_[:, :3] - feat_center
        feat_ = np.concatenate((feat_, feat_center_offset), axis=1)
        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)


        pc = pc_[inds]
        feat = feat_[inds]
        pc_data = SparseTensor(feat, pc)

        return {
            'lidar': pc_data,
            "image": img,
            'targets': label
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
