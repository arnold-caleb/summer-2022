import os
import torch
import time
import numpy as np

from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop
import torch.nn.functional as F

from models.unconditional_unet import Unet # Unet with Attention (Linear Attention and Residual Connections)

batch_size = 8

data_transforms = {x: transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224),
        transforms.RandAugment(), # https://arxiv.org/abs/1909.13719
        # transforms.AugMix(), # https://arxiv.org/abs/1912.02781
        transforms.RandomInvert(),
        transforms.RandomPosterize(bits=2),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.TrivialAugmentWide(),
        transforms.RandomEqualize(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor()]) 
    for x in ['train', 'val']}

image_datasets = datasets.ImageFolder('../binary_classifier_a', data_transforms['train'])
train_set, val_set = torch.utils.data.random_split(image_datasets, [round(len(image_datasets) * 0.75), round(len(image_datasets) * 0.25)])

dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)}

#class_names = image_datasets.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.nn.Sequential(
    Unet(dim=224, channels=3, dim_mults=(1, 2, 4,)),
    torch.nn.Flatten(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(150528, 2)
)

criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=1e-5)
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

def train_loop(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        pred = model(images.to(device))
        loss = criterion(pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images.to(device))
            test_loss += criterion(pred, labels.to(device)).item()
            correct += (pred.argmax(1) == labels.to(device)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 200 
since = time.time()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloaders['train'], model.to(device), criterion, optimizer)
    test_loop(dataloaders['val'], model.to(device), criterion)

time_elapsed = time.time() - since
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print("Done!")

torch.save(model.state_dict(), 'classification_weights_atelectasis.pth')




