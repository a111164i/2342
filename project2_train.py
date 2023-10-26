'''
this script is for the training code of Project 2..

-------------------------------------------
INTRO:
You can change any parts of this code

-------------------------------------------

NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email:
txue4133@uni.sydney.edu.au, weiyu.ju@sydney.edu.au
'''

# import the packages
import argparse
import logging
import math
import multiprocessing
import sys
import time
import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import network as Network
from network import combined

parser = argparse.ArgumentParser(description= \
                                     'scipt for training of project 2')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Used when there are cuda installed.')
args = parser.parse_args()

# Check for GPU availability, let model run on GPU if possible
print("PyTorch Version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)


##############################################
# training process.
def train_net(net, train_loader, val_loader, opter, crit, num_epochs, device):
    best_val_acc = 0.0
    print_step = 150
    model_path = 'project2.pth'

    if device == 'cuda':
        net = net.cuda()

    for epoch in range(num_epochs):
        net.train()  # Set the model to training mode
        running_loss = 0.0
        trained = 0  # used to count trained images in each epoch
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if device == 'cuda':
                inputs, labels = inputs.cuda(), labels.cuda()

            trained += 1
            if trained % print_step == 1:
                print(f'{trained} / {len(train_loader)} ' + f' epoch: {epoch}')

            opter.zero_grad()
            outputs = net(inputs)
            loss = crit(outputs, labels)
            loss.backward()
            opter.step()

            running_loss += loss.item()

        # Calculate training loss for the epoch
        avg_loss = running_loss / len(train_loader)

        # Validation phase per epoch
        net.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                if device == 'cuda':
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {avg_loss:.4f} - Validation Accuracy: {val_acc:.2f}%')

        # Save the best model in epochs
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), model_path)

    print("Training complete.")
    return best_val_acc


##############################################
# Transformation definition
# NOTE:
# Write the train_transform here. We recommend you use
# Normalization, RandomCrop and any other transform you think is useful.
if __name__ == '__main__':
    # define transform
    # add data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip vertically
        transforms.RandomRotation(degrees=30),  # Randomly rotate by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define the training dataset and dataloader.
    image_path = './5307Project2'
    # Load the entire dataset
    allset = ImageFolder(image_path, train_transform)
    # Get labels from the dataset
    labels = [sample[1] for sample in allset]

    # Define the ratio of data to allocate for the validation set.
    validation_ratio = 0.2  # 20% of the data will be used for validation

    # Split allset into training and validation sets
    train_indices, val_indices = train_test_split(
        range(len(allset)), test_size=validation_ratio, random_state=42)

    batch_size = 4

    # Create training and validation subset from allset using indices created before
    trainset = torch.utils.data.Subset(allset, train_indices)
    valset = torch.utils.data.Subset(allset, val_indices)

    # Create tensor data for training and validation subset
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size, shuffle=True, num_workers=2)

    # initialize to_train_net
    # net_1, net_1_input = Network.efficientnet()
    # net_2, net_2_input = Network.vit()
    # net_3, net_3_input = Network.alexnet()
    # to_train_net = Network.combined(net_1, net_2, net_3, net_1_input, net_2_input, net_3_input, 32)
    my_classifier = Network.my_classifier(batch_size, 32)

    # define hyperparameters and train the network here
    learning_rate = 0.0001
    num_epochs = 16
    optimizer = torch.optim.AdamW(my_classifier.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = train_net(my_classifier, trainloader, valloader, optimizer, criterion, num_epochs, device)

    print("final validation accuracy:", best_acc)
