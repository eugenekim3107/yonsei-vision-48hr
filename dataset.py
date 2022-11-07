import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class cifarDataset(Dataset):
    def __init__(self, csv, dir_name, transform=None):
        self.annotations = pd.read_csv(os.path.join(dir_name, csv))
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
        label = torch.tensor(self.word_to_num[self.annotations.iloc[index, 1]], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    data = cifarDataset(csv="data/cifar100_nl_clean.csv",
                        dir_name="dataset",
                        transform=transforms.ToTensor())
    batch_size = 1
    train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    for (image, label) in train_loader:
        print(image.shape, label)
        plt.imsave("test_img.jpg", image[0].permute(1, 2, 0).numpy())
        break

if __name__ == '__main__':
    main()
