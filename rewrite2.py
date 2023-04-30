from sklearn.neural_network import MLPClassifier
import numpy as np

import os
import random
from PIL import Image
import numpy as np


listOfImages = os.listdir("newData")
random.shuffle(listOfImages)

def mean(values):
    x = sum(values)/len(values)
    return x

def get_letter_activation(filename):
    prepath = os.getcwd()
    path = f"{prepath}\\newData\\{filename}"
    img = Image.open(path)
    
    greyvalues = [mean(x) for x in img.getdata()]
    x = greyvalues
    greyvalues = (x-np.min(x))/(np.max(x)-np.min(x))
    greyvalues = [round(xs,2) for xs in greyvalues]
    return greyvalues



input_vectors_list = []
targets_list = []


for f in listOfImages:
    
    greyvalues = get_letter_activation(f)
    input_vectors_list.append(greyvalues)
    df = [0]*27
    
    listOfLetters = ["-a", "-b", "-c", "-d", "-e", "-f", "-g", "-h", "-i", "-j", "-k", "-l", "-m", "-o", "-p", "-q", "-r", "-s", "-t", "-u", "-v", "-w", "-x", "-y", "-z"]
    
    if "cap-" in f:
        df[26]=1
    else: df[26]=0
        
    for i,v in enumerate(listOfLetters):
        if v in f:
            df[i]=1
            break
    targets_list.append(df)
    


X = input_vectors_list
X = np.nan_to_num(X)
y = targets_list

clf = MLPClassifier(hidden_layer_sizes=(100,), 
                    activation='relu', 
                    solver='adam', 
                    alpha=0.0001, 
                    batch_size='auto', 
                    learning_rate='constant', 
                    learning_rate_init=0.001, 
                    power_t=0.5, 
                    max_iter=500, 
                    shuffle=True, 
                    random_state=None, 
                    tol=0.0001, 
                    verbose=True, 
                    warm_start=False,
                    momentum=0.9, 
                    nesterovs_momentum=True, 
                    early_stopping=False, 
                    validation_fraction=0.1, 
                    beta_1=0.9, beta_2=0.999, 
                    epsilon=1e-08, 
                    n_iter_no_change=10, 
                    max_fun=15000)

print(X[0])

clf.fit(X, y)

print(clf.predict(
    [get_letter_activation("cap-a-0.png")]
))  