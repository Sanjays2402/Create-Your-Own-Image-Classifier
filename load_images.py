import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms 

def load_images(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    #Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], 
                                                       [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) 
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms) 
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    #Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_dataset, validation_dataset, test_dataset, trainloader, validloader, testloader