import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from data.utils import (
    binary_loader,
    collect_r2c_data,
    colorEnhance,
    cv_random_flip,
    randomCrop,
    randomPeper,
    randomRotation,
    rgb_loader,
)


class RefCODDataset(Dataset):
    def __init__(self, data_root, mode='train', shot=5, image_size=384):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_root = data_root
        self.shot = shot
        self.data_list, self.class_file_list = collect_r2c_data(data_root=self.data_root, mode=self.mode)

        if self.mode == 'val' and self.shot not in [-1, 0, 5]:
            self._record_val_support_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        sample_name = image_path.split('/')[-1][:-4]

        image = rgb_loader(image_path)
        label = binary_loader(label_path)

        if self.mode == 'train':
            image, label = self._augment(image=image, label=label)

        image = self.img_transform(image)
        if self.mode == 'train':
            label = self.gt_transform(label)
        else:
            label = np.asarray(label, np.float32)

        support_feature = self._load_support_feature(image_path)
        return image, label, support_feature, sample_name

    def _load_support_feature(self, image_path):
        if not (self.shot > 0 or self.shot == -1):
            return -1

        class_name = image_path.split('/')[-1].split('-')[-2]
        candidate_files = self.class_file_list[class_name]
        num_candidates = len(candidate_files)

        if self.mode == 'train':
            chosen_indices = random.sample(range(num_candidates), self.shot) if self.shot > 0 else list(range(num_candidates))
        else:
            chosen_indices = list(range(num_candidates))

        feature_list = []
        for idx in chosen_indices:
            feature = np.load(candidate_files[idx])
            feature_list.append(torch.from_numpy(feature))

        return sum(feature_list) / len(feature_list)

    def _record_val_support_files(self):
        file_path = f'./data/dataset_{self.shot}shot_val.json'

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.class_file_list = json.load(f)
            return

        for cate, feature_files in self.class_file_list.items():
            assert len(feature_files) > self.shot
            rand_idxs = random.sample(range(len(feature_files)), self.shot)
            self.class_file_list[cate] = [feature_files[idx] for idx in rand_idxs]

        with open(file_path, 'w') as f:
            json.dump(self.class_file_list, f, indent=4)

    def _augment(self, image, label):
        image, label = cv_random_flip(image, label)
        image, label = randomCrop(image, label)
        image, label = randomRotation(image, label)
        image = colorEnhance(image)
        label = randomPeper(label)
        return image, label
