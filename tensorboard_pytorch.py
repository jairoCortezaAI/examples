import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
from torch import nn
import matplotlib.pyplot as plt
#importing tensorboard
from torch.utils.tensorboard import SummaryWriter

#define el dispositivo
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

#Tensorboard summary writer
writer = SummaryWriter()

#DEFINICION DE NN
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

#DEFINICION DE ITERACIONES
def train_loop(data_loader, nn_model,loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X,y) in enumerate(data_loader):
            #Device
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
                #Send information to tensorboard writer
                writer.add_scalar('Loss',loss, batch)

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
#definir datos
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


#ENTRENAMIENTO
model=NeuralNetwork()
model.to(device)

print(model)

learning_rate = 1e-3
epochs = 4
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr = learning_rate)

for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader,model,loss_function,optim)
        final_acc=test_loop(test_dataloader,model,loss_function)
        writer.add_scalar('Accuracy/test',final_acc,t)
print("Finished")

#Close writer
writer.flush()
writer.close()
