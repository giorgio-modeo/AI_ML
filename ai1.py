#inteligenza artificiale
import numpy as np
import nn1
import matplotlib.pyplot as plt
input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult
dot_product_1 = np.dot(input_vector, weights_1)
print(f"The dot product is: {dot_product_1}")
dot_product_2 = np.dot(input_vector, weights_2)
print(f"The dot product is: {dot_product_2}")
input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])
def sigmoid(x):
       return 1 / (1 + np.exp(-x))
def make_prediction(input_vector, weights, bias):
      layer_1 = np.dot(input_vector, weights) + bias
      layer_2 = sigmoid(layer_1)
      return layer_2
prediction = make_prediction(input_vector, weights_1, bias)
print(f"The prediction result is: {prediction}")
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