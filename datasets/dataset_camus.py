import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly rotate by 0,90,180,270 then flip horizontally/vertically."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size: List[int]):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ValTransform(object):
    def __init__(self, output_size: List[int]):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long()}


class CamusDataset(Dataset):
    """CAMUS 2D echocardiography dataset loader using list files of image/mask pairs."""

    def __init__(self, root_path: str, list_path: str, split: str = "train", transform=None):
        self.root_path = root_path
        self.list_path = list_path
        self.split = split
        self.transform = transform
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip().split() for line in f.readlines() if line.strip()]
        if not self.sample_list:
            raise ValueError(f"Empty list file: {list_path}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_rel, label_rel = self.sample_list[idx]
        image_path = os.path.join(self.root_path, image_rel)
        label_path = os.path.join(self.root_path, label_rel)

        image = np.array(Image.open(image_path).convert('L'), dtype=np.float32)
        label = np.array(Image.open(label_path).convert('L'), dtype=np.int64)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.splitext(os.path.basename(label_rel))[0]
        return sample
