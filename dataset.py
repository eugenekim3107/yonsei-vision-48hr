import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
import pandas as pd

class cifarDataset(Dataset):
    def __init__(self, csv, dir_name, transform=None):
        self.annotations = pd.read_csv(csv)
        self.dir_name = dir_name
        self.transform = transform
        self.num_to_word = dict(enumerate(self.annotations.iloc[:,1].unique()))
        self.word_to_num = {value:key for (key,value) in self.num_to_word.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = self.annotations.iloc[index, 0]
        image = cv2.imread(os.path.join(self.dir_name, image_path))
        image = cv2.resize(image, (32,32))
        label = torch.tensor(self.word_to_num[self.annotations.iloc[index, 1]], dtype=torch.uint8)
        if self.transform:
            image = self.transform(image)
        return image, label
