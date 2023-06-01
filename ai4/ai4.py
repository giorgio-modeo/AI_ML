# input
# semplifica l'input togliendo le cogniugazzioni
# suddividere la domanda dalle specifiche 
# cercare il topic (es. ricetta con le patate)
#                 topic = ricetta
#                 specifiche = patate
# analizzare la frase con la self attention(percentuali di pertinenza alla domanda)
# reitera il risultato in modo da perfezzionare la risposta
# query
# risposta
import nn1
learning_rate = 0.1
neural_network = nn1.NeuralNetwork(learning_rate)



testo = "testo prima prova"
testo_diviso = testo.split(" ")
print(testo_diviso)

for i in testo_diviso:
    if i == ("spiega",):
        domanda = i


nn1.

