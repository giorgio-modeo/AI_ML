import sys
sys.path.append("nn1")
import nn2
import numpy as np
import matplotlib.pyplot as plt
learning_rate = 0.1
# input_vector = np.array([2, 1.5])
# print(input_vector)
# neural_network = nn2.NeuralNetwork(learning_rate)
# neural_network.predict(input_vector)
# [2.  1.5]

# gli imput devono essere proporzionali ai target
input_vectors = np.array([[3, 1.5],[2, 1],[4, 1.5],[3, 4],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1],[26,1],])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, ])
print(input_vectors)
# [[ 3.   1.5]
#  [ 2.   1. ]
#  [ 4.   1.5]
#  [ 3.   4. ]
#  [ 3.5  0.5]
#  [ 2.   0.5]
#  [ 5.5  1. ]
#  [ 1.   1. ]
#  [26.   1. ]]

neural_network = nn2.NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
print(neural_network.predict(input_vectors))
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("ai3/cumulative_error.png")