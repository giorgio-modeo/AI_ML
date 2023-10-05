import nn3 
learning_rate = 0.1
neural_network = nn3.NeuralNetwork(learning_rate)

frase = 'ciao sono Giorgio e mi piacciono i treni'
target= 'ciao Giorgio anche a me piacciono i treni'
terr = nn3.train(frase, target,30)
print(terr)

