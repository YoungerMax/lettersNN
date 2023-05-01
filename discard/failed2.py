import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from PIL import Image
# load the dataset, split into input (X) and output (y) variables




listOfImages = os.listdir("newData")
random.shuffle(listOfImages)







def mean(values):
    x = sum(values)/len(values)
    
    return x

prepath = os.getcwd()


input_vectors_list = []
targets_list = []


for f in listOfImages:
    
    path = f"{prepath}\\newData\\{f}"
    img = Image.open(path)
    
    greyvalues = [mean(x) for x in img.getdata()]
    x = greyvalues
    greyvalues = (x-np.min(x))/(np.max(x)-np.min(x))

    input_vectors_list.append(greyvalues)
    
    df = [0]*27
    
    listOfLetters = ["-a", "-b", "-c", "-d", "-e", "-f", "-g", "-h", "-i", "-j", "-k", "-l", "-m", "-o", "-p", "-q", "-r", "-s", "-t", "-u", "-v", "-w", "-x", "-y", "-z"]
    
    if "cap-" in f:
        df[26]=1
    else: df[26]=0
        
    for i,v in enumerate(listOfLetters):
        if v in f:
            df[i]=1
            targets_list.append(df)
            break



X = input_vectors_list
y = targets_list

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
print(X)
print(y)
# define the model
model = nn.Sequential(
    nn.Linear(784, 784),
    nn.ReLU(),
    nn.Linear(784, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")