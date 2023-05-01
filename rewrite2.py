from sklearn.neural_network import MLPClassifier
import numpy as np

import os
import random
from PIL import Image


listOfImages = os.listdir("newData")
listOfLetters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
os.chdir("newData")

random.shuffle(listOfImages)

def mean(values: int) -> float:
    x = sum(values)/len(values)
    return x

def get_letter_activation(path:str) -> list:
    img = Image.open(path)
    
    gv = [mean(x) for x in img.getdata()]
    gv = (gv-np.min(gv))/(np.max(gv)-np.min(gv))
    gv = [round(xs,2) for xs in gv]
    return gv

def confirm_output(output: np.array, letter: str) -> bool:
    output = output[0].tolist()
    assumed_letter = listOfLetters[output.index(1)].lower()
    if output[-1]:
        assumed_letter = assumed_letter.upper()
    return [assumed_letter, assumed_letter == letter]
    



input_vectors_list = []
targets_list = []


for f in listOfImages:
    input_vectors_list.append(get_letter_activation(f))
    df = [0]*27
    
    
    if "cap-" in f:
        df[26]=1
        
    df[listOfLetters.index(f[4])] = 1
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



prediction = clf.predict(
    [get_letter_activation("cap-q-5.png")] 
)
print(prediction)

print(confirm_output(prediction, "Q"))



