import torch
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ResNet18
import os
import pandas as pd
from dataset import cifarDataset
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def weightedCSV(original_csv, dir_name, supervisor_model, sample_num):

    csv_file = pd.read_csv(os.path.join(dir_name, original_csv), header=None)
    dataset = cifarDataset(csv=original_csv, dir_name=dir_name,
                           transform=transforms.ToTensor())
    batch_size = 1
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervisor_model.eval()
    supervisor_model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    file_names = []
    labels = []
    proportions = []
    with torch.no_grad():
        for (x, y, image_path) in loader:
            x, y = x.to(device), y.to(device)
            out = supervisor_model(x)
            loss = 1. - loss_fn(out, y)
            proportions.append(float(loss))
            labels.append(int(y))
            file_names.append(image_path[0])

    # Normalize proportions
    proportions = (np.array(proportions) - min(proportions)) / max(proportions)
    proportions = proportions / np.sum(proportions)
    # Sample from newly created distribution
    new_data = np.random.choice(file_names, sample_num, p=proportions)
    final_csv = pd.DataFrame(columns=[0, 1])
    for i in new_data:
        final_csv = pd.concat([final_csv, csv_file[csv_file[0] == i]],
                              ignore_index=True)
    final_csv.to_csv("dataset/data/cifar100_weighted.csv", index=False)

def main():
    model = torch.load("modelStates/supervisorModel.pth")

if __name__ == "__main__":
    main()

