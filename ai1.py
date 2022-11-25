#inteligenza artificiale
#importo numpy per i calcoli complessi
import numpy as np

import nn1

import matplotlib.pyplot as plt

################################################################################################################### questo segmento e per fare un esempio 

#inmetto i punti dei vettori
input_vector = [1.72, 1.23]

#inposto i pesi per far in modo che successivamente il prodotto non possa venire 0
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# calcolo il prodotto 
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult
dot_product_1 = np.dot(input_vector, weights_1)
dot_product_2 = np.dot(input_vector, weights_2)

#stamp di verifica
print(f"The dot product is: {dot_product_1}")
print(f"The dot product is: {dot_product_2}")
###################################################################################################################


################################################################################################################### secondo segmento di prova

#inmetto i punti dei vettori 
          #    np.array() serve pe dichiarare un array di numeri
input_vector = np.array([1.66, 1.56,])
#inposto i pesi
weights_1 = np.array([1.45, -0.66])

#paragone stupido ma efficace "i bias sono i neuroni del'ai", piu il valore del bias sara basso piu esso si attivera prima 
"""
      lento
      |     /
      |    /
      |   /
      |  /
      | /
veloce /____________________
      0                     n
                                                                           cambia la predizzione il margine di errore che e molto inferiore
----------bias -1.0
The prediction result is: [0.34919035]
The prediction result is: [0.47751518]
Prediction: [0.47751518]; Error: [0.22802074]
The derivative is [0.95503035]
Prediction: [0.03129183]; Error: [0.00097918]

----------bias 0 0 
The prediction result is: [0.7985731]
The prediction result is: [0.87101915]
Prediction: [0.87101915]; Error: [0.75867436]
The derivative is [1.7420383]
Prediction: [0.01496248]; Error: [0.00022388]
obbligatorio mttere la virgola
"""
bias = np.array([0.0])
bias2 = np.array([-1.0])

#    e una funzione non lineare (e stabile su un valore poi incrementa o decrementa per un lesso di tempo ben definito e infine si stabilizza su un altro valore)
#    nello specifico la sigmoide passa da (-n a n) sull asse delle x mentre sull asse delle y passa da 0 a 1 
#    proprio per questo viene utilizzata nelle ai OGNI numero che entra diventera 1
def sigmoid(x):
       return 1 / (1 + np.exp(-x))
#    creo l'algoritmo della predizione
def make_prediction(input_vector, weights, bias):
     #    creo i layer per fare le predizzioni, servono per far comunicare le diverse fasi del processo 
          #layer di input/hiden
      layer_1 = np.dot(input_vector, weights) + bias
      layer_2 = np.dot(input_vector, weights) + bias2
          #layer di output
      layer_3 = sigmoid(layer_1 + layer_2)
      return layer_3

#    faccio la predizzione con i vettori standard
prediction = make_prediction(input_vector, weights_1, bias, bias2)
print(f"The prediction result is: {prediction}")

#    faccio la predizzione modificando i vettori 
input_vector = np.array([2, 1.5])
prediction = make_prediction(input_vector, weights_1, bias)
print(f"The prediction result is: {prediction}")

target = 0
mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")

derivative = 2 * (prediction - target)
print(f"The derivative is {derivative}")

weights_1 = weights_1 - derivative
prediction = make_prediction(input_vector, weights_1, bias)
error = (prediction - target) ** 2
print(f"Prediction: {prediction}; Error: {error}")

def sigmoid_deriv(x):
     return sigmoid(x) * (1-sigmoid(x))
derror_dprediction = 2 * (prediction - target)
layer_1 = np.dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1
derror_dbias = (
     derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
 )
 #
learning_rate = 0.1
neural_network = nn1.NeuralNetwork(learning_rate)
neural_network.predict(input_vector)
input_vectors = np.array([[3, 1.5],[2, 1],[4, 1.5],[3, 4],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1],])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
neural_network = nn1.NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")