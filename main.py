import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random




class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()]*392)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
    
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = mean(2 * (prediction - target))
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        print(dprediction_dlayer1)
        print(derror_dprediction)
        print(dlayer1_dbias)
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                #input_vectors.reshape(27,1870)
                print(input_vectors.shape)

                for data_instance_index in range(len(input_vectors)):
                    print(data_instance_index)
                    data_point = input_vectors[data_instance_index]
                    new_targets = targets.reshape(1870,27)#.tolist()

                    
                    


                    target = new_targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
        

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
    
    greyvalues = np.array([mean(x) for x in img.getdata()])
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



input_vectors = np.array(
    input_vectors_list
) 



targets = np.array(
    targets_list
)



learning_rate = 0.001

neural_network = NeuralNetwork(learning_rate)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")
