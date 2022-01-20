import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
from torch import nn
from mlflow import log_metric, log_param, log_artifacts

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

def train_loop(data_loader, nn_model,loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X,y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            #Predecir y calcular error
            pred = nn_model(X)
            loss = loss_fn(pred,y)

            #Retropropagacion
            #Volver cero los valores de gradiente
            optimizer.zero_grad()
            #propagar el error
            loss.backward()
            #Ajustar parametros
            optimizer.step()

            if batch%100 ==0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(data_loader,nn_model,loss_fn):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            #Device
            X, y = X.to(device), y.to(device)
            pred = nn_model(X)
            test_loss +=loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /=num_batches
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct*100 


training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform=ToTensor()#,
    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download = True,
    transform = ToTensor()#,
    #target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#Dataloaders
train_dataloader=DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

model=NeuralNetwork().to(device)
print(model)

#PARAMETERS
learning_rate = 1.2e-3
epochs = 5

loss_function = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr = learning_rate)

for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader,model,loss_function,optim)
        final_acc = test_loop(test_dataloader,model,loss_function)
print("Finished")

if __name__ == "__main__":
    #log a parameter
    log_param("Learning rate",learning_rate)
    log_param("Epochs",epochs)

    #log metrics
    log_metric("Accuracy", final_acc)