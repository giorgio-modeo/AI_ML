import sys
sys.path.append("nn1")
import nn1
import numpy as np
import matplotlib.pyplot as plt
import os,time
#     non so cosa succede qui
# input_vector = np.array([2, 1.5])
# print(input_vector)
learning_rate = 0.01
# neural_network = nn1.NeuralNetwork(learning_rate)
# a =neural_network.predict(input_vector)

# print(a)
# a = a*10
# print(a,"\n")

# for c in range(5):
#     if a<65:
#         a= a*10
#     elif a>90:
#         a= a/10
#     else:
#         break
# if a>90:
#     a= a/10

# print(chr(int(a)))

# [2.  1.5]

# gli imput devono essere proporzionali ai target

input_vectors = np.array([[3, 1],[2, 1],[4, 1],[3, 4],[3, 0],[2, 0],[5, 1],[1, 1],[26,1],])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, ])

i=0
os.system('cls')
neural_network = nn1.NeuralNetwork(learning_rate)
a =neural_network.predict(input_vectors)
for d in a:
    i+=1
    print("\n-------------------- ",i)
    while True:
        if d<65:
            d *= 10
        else:
            d=int(d)
            if d>122:
                print("null")
            elif d>=97:
                print(chr(d))
            elif d>90 and d<96:
                print("null")
            print(d)
            break
print("\n--------------------")
training_error = neural_network.train(input_vectors, targets, 10000)
