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


__all__ = ['SketchImageData']

class SketchImageData(dict):

    def __init__(self):

        super().__init__({
            'train':
                SketchImage(split='train'),
            'test':
                SketchImage(split='val')})


class SketchImage:

    def __init__(self, split):
        self.split = split
        self.data = []
        self.label = []
        partition = split
        with open(f"pointcloudsxy/tiny_{partition}_set.txt", "r") as f:
            for lines in f.readlines():
                line_list = lines.strip("\n").split(" ")
                self.data.append(os.path.join(f"pointclouds_xy/{partition}", line_list[0]))
                self.label.append(int(line_list[1]))
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_size = [256, 256]
        # input_size = [224, 224]
        self.mode = partition
        if partition == "train":
            self.transform = T.Compose([
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy("imagenet")),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize(input_size),
                T.ToTensor(),
                normalize
                # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])

        else:
            self.transform = T.Compose([
                T.Resize(input_size),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = Image.open(self.data[index].replace("pointclouds_xy", "/media/kemove/403/TEI").replace(".txt", ".png"))
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(self.label[index], dtype=torch.long)

        return {
            "image": img,
            'targets': label
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
