import torch
import torch.optim as optim
import torchvision.transforms.functional as FT
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ResNet18
import os
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

def main():
    torch.cuda.empty_cache()
    model = ResNet18()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    dataset = CIFAR100(download=True,root="./trueCIFAR100",transform=transform)
    train_set,_ = torch.utils.data.random_split(dataset, [5000, 45000])
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=False)

    for epoch in range(epochs):
        train_fn(train_loader, model, optimizer, loss_fn)
    torch.save(model.state_dict(), "modelStates/supervisorModel.pth")

if __name__ == "__main__":
    main()


