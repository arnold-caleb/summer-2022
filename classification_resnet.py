import os
import torch
import time
import numpy as np
import copy
import wandb # to visualize training within the ML project

from einops import rearrange
from models.unconditional_unet import Unet

import torch.nn.functional as F
from torch import optim
from torch.quantization import convert

from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop
import torchvision.models.quantization as models

from models.attention import Attention, LinearAttention
from models.resnets import Residual, PreNorm

wandb.init(project="normal_atelectasis (attention)") 
batch_size = 32 

data_transforms = {x: transforms.Compose([
        transforms.Resize(128), 
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandAugment(),              # https://arxiv.org/abs/1909.13719
        # transforms.AugMix(),                 # https://arxiv.org/abs/1912.02781
        transforms.RandomInvert(),
        transforms.RandomPosterize(bits=2),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.TrivialAugmentWide(),
        transforms.RandomEqualize(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor()]) 
    for x in ['train', 'val']}

image_datasets = datasets.ImageFolder('../binary_classifier_a', data_transforms['train'])
train_set, val_set = torch.utils.data.random_split(image_datasets, 
        [round(len(image_datasets) * 0.75), round(len(image_datasets) * 0.25)])

dataloaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
        "val": torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)}

# class_names = image_datasets.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model = models.resnet50(pretrained=True, progress=True, quantize=False)
num_ftrs = model.fc.in_features

def create_combined_model(model):
  # pretrained resnet50 feature extractor.
  model_features = torch.nn.Sequential(
    model.quant,  # Quantize the input
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
    model.avgpool,
    model.dequant,  # Dequantize the output
  )

  # Binary Classifier Head 
  new_head = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(num_ftrs, 2),)

  # Combine along with the quant stubs and the attention mechanism.
  new_model = torch.nn.Sequential(
          Residual(PreNorm(3, LinearAttention(dim=3, heads=3))),
          model_features, # pretrained (makes training much easier) 
          Residual(PreNorm(2048, LinearAttention(dim=2048, heads=3))),
          torch.nn.Flatten(1),
          new_head,)

  return new_model

model.train()
model.fuse_model()

model = create_combined_model(model)
model[0].qconfig = torch.quantization.default_qat_qconfig
model = torch.quantization.prepare_qat(model, inplace=True)

for param in model.parameters():
    param.requires_grad = True

model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

learning_rate = 1e-5
epochs = 150 
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.load_state_dict(torch.load("classifcation_resnet_weights_atelectasis_attention.pth"))

# train
def train_loop(dataloader, model, criterion, optimizer):

    model.train()
    size = len(dataloader.dataset)
    for batch, (images, labels) in enumerate(dataloader):
        pred = model(images.to(device))
        loss = criterion(pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# val
def test_loop(dataloader, model, criterion, best_acc, best_model_wts):

    model.eval()
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

    if correct > best_acc:
        best_acc = correct
        best_model_wts = copy.deepcopy(model.state_dict())
        wandb.alert(
                title="New High Accuracy", text=f"Accuracy {100 * best_acc} is the new high", \
                level=wandb.AlertLevel.INFO)

    wandb.log({"test_loss": test_loss, "epoch accuracy": 100 * correct, "overall accuracy": 100 * best_acc})

    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Best Accuracy: {(best_acc * 100):>0.1f}% \n")

    return best_acc, best_model_wts

wandb.watch(model)
wandb.config = {
        "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, 
        "classes": image_datasets.classes}

# keep track of the best model weights and accuracy
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

since = time.time()


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloaders['train'], model.to(device), criterion, optimizer)
    best_acc, best_model_wts = test_loop(dataloaders['val'], model.to(device), criterion, 
                                           best_acc, best_model_wts)

time_elapsed = time.time() - since

print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print("Done!")

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'classifcation_resnet_weights_atelectasis_attention.pth')

