import nn3 
import gensim.models.word2vec as KeyedVectors2
learning_rate = 0.1
neural_network = nn3.NeuralNetwork(learning_rate)

sentences = ["Ciao, come stai?", "Sto bene, grazie. E tu?", "Anch'io sto bene."]
# Crea un nuovo modello Word2Vec
model = KeyedVectors2.Word2Vec(sentences, vector_size=100, window=5, min_count=1)
# Salva il modello
model.save("word2vec.model")

frase = 'ciao sono Giorgio e mi piacciono i treni'
target= 'ciao Giorgio anche a me piacciono i treni'
terr = neural_network.train(frase, target,30)
print(terr)

