import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
#import helper
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

from cae_32x32x32 import CAE

#TO DO:
# - Fix and improve training
# - Gather more training data
# - Fix/normalize output
def define_model():
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device.type)
    model = CAE().to(device)

    # create an optimizer object
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # MSE Loss function
    criterion = nn.MSELoss()
    return model, optimizer, criterion, device

def load_data(root='data', type='train_images'):
    path = root + "/" + type
    #apply a random crop transform to obtain a 3x128x128 image to feed into CAE model
    transform = transforms.Compose([transforms.RandomCrop(128),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()
                                   ])
    #Load local dataset from image folder
    dataset = datasets.ImageFolder(path, transform=transform)
    #TO DO: get more training data an increase batch size
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    #showImages(data_loader)
    return data_loader

'''
showImages(data_loader)
Used to show images from the dataloader to insure proper loading
In the running of the model, this function will not be used
'''

def showImages(data_loader):
    images = next(iter(data_loader))
    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for i in range(4):
        ax = axes[i]
        # takes 4 different random crops from 4 different image
        axes[i] = imshow(images[0][i],i,ax=ax, normalize=False)
    plt.show()

'''
imshow(image,ax, normalize)
Based on: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/helper.py
Helper function for showImages to display tensor images using matplotlib
'''
def imshow(image, ax=None, title=None, normalize=True):
    #Imshow for Tensors
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy()
    image = image.transpose((1, 2, 0)) #RGB channel needs to be permuted to the end for matplotlib

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


def load_model(PATH):
    model = CAE()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    return model

def train_model(train_loader,model_list):
    model, optimizer, criterion, device = model_list
    model.train()
    for epoch in range(5):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [-1, 3x128x128] patches
            # load it to the active device
            batch_features = batch_features.view(-1, 3, 128, 128).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}, loss = {:.6f}".format(epoch + 1, loss))
    return model

def test(trained_model,test_loader):
    test_examples = None
    trained_model.eval()
    with torch.no_grad():
        for batch_features in test_loader:
            test_examples = batch_features[0].view(-1,3,128,128)
            print(test_examples.size())
            reconstruction = trained_model(test_examples)
            break
    with torch.no_grad():
        number = 7
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].permute(1,2,0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].permute(1,2,0))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

if __name__=="__main__":
    model_list = define_model()
    saved_model = load_model('saved_models/model_yt_small_final.state')
    saved_model_list = (saved_model, model_list[1], model_list[2], model_list[3])
    data_loader = load_data()
    trained_model = train_model(data_loader, model_list)
    test_loader = load_data() #load different shuffle of the dataset (will be changed to a different test set)

    test(saved_model,test_loader)