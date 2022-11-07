import torch
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ResNet18
import os
from dataset import cifarDataset
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
lr = 2e-5
batch_size = 1000
weight_decay = 1e-4
epochs = 30
transform = transforms.ToTensor()
train_accuracy = []
test_accuracy = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fn(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)

    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loop):
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += int(sum(out.argmax(axis=1) == y))
        total += y.size(0)

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

    accu = 100. * (correct / total)
    train_accuracy.append(accu)


def test_fn(test_loader, model):
    model.eval()
    loop = tqdm(test_loader)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x,y = x.to(device), y.to(device)
            out = model(x)
            correct += int(sum(out.argmax(axis=1) == y))
            total += y.size(0)

    accu = 100. * (correct / total)
    test_accuracy.append(accu)


def main():
    torch.cuda.empty_cache()
    model = ResNet18()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dir_name = "dataset"
    csv = "trueCIFAR100/cifar100_nl_clean.csv"
    csv_test = "trueCIFAR100/cifar100_nl_test.csv"
    dataset = cifarDataset(csv=csv, dir_name=dir_name, transform=transform)
    test_set = cifarDataset(csv=csv_test, dir_name=dir_name,
                            transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 9998])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=False)

    for epoch in range(epochs):
        train_fn(train_loader, model, optimizer, loss_fn)
        test_fn(test_loader, model)
    torch.save(model.state_dict(), "modelStates/baseModel.pth")

if __name__ == "__main__":
    main()
