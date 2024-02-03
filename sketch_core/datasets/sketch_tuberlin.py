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


__all__ = ['SketchTuberlinImageData']

class SketchTuberlinImageData(dict):

    def __init__(self):

        super().__init__({
            'train':
                SketchTuberlinImage(split='train'),
            'test':
                SketchTuberlinImage(split='val')})


class SketchTuberlinImage:

    def __init__(self, split):
        self.split = split
        self.data = []
        self.label = []
        partition = split
        if partition == "train":
            with open("ntuberlin/train0.txt", "r") as f:
                for lines in f.readlines():
                    line_list = lines.strip("\n").split("png ")
                    self.data.append(line_list[0]+"png")
                    self.label.append(int(line_list[1]))
        else:
            with open("ntuberlin/test0.txt", "r") as f:
                for lines in f.readlines():
                    line_list = lines.strip("\n").split("png ")
                    self.data.append(line_list[0]+"png")
                    self.label.append(int(line_list[-1]))
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_size = [256, 256]
        if partition == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                # T.RandomRotation15),(
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # img = Image.open(self.data[index])
        img = cv2.imread(self.data[index])
        img = cv2.resize(img, (256, 256), dst=None)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = torch.tensor(self.label[index], dtype=torch.long)
        # img = Image.fromarray(255 - np.array(img))
        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            'targets': label
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
