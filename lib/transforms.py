import os
import torch
from torchvision import transforms, datasets

def data_transforms():

    #data_dir = "../experiments/data/cifar10"
    data_dir = "../Upload/Chest-Xray"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
            'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
            'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])}

    dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=8)
            for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names 

