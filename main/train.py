from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

def train_model(model,trainloader,epochs,device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in tqdm(range(epochs)):
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
    return model

def eval_model(model,testloader,device="cuda"):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)              
            preds = torch.argmax(outputs, dim=1)  

            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return accuracy