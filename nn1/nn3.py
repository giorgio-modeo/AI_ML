import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import gensim.models.word2vec as KeyedVectors2


# Crea un set di dati di frasi
sentences = ["Ciao, come stai?", "Sto bene, grazie. E tu?", "Anch'io sto bene."]


# Crea un nuovo modello Word2Vec
model = KeyedVectors2.Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Salva il modello
model.save("word2vec.model")

# Definizione della classe NeuralNetwork
class NeuralNetwork:
    # Definizione del metodo init, che inizializza i pesi, il bias e il learning rate
    def __init__(self, learning_rate):
        # Inizializzazione dei pesi con i vettori di embedding del modello Word2Vec

        self.weights = KeyedVectors.load_word2vec_format("word2vec.model")
        self.weights_2 = KeyedVectors.load_word2vec_format("./nn1/word2vec.model")

        self.bias = np.random.normal(loc=0.0,scale=0.01,size=(100,200))
        self.bias_2 = np.random.normal(loc=0.0,scale=0.01,size=(100,200))
        # Impostazione del learning rate
        self.learning_rate = learning_rate
    
    # Definizione della funzione di attivazione sigmoid
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Definizione della derivata della funzione di attivazione sigmoid
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def tokenize(text):
        return text.split()
    
    def word2veclookup(model, token):
    # Controlla se il token esiste nel modello Word2Vec
        if token not in model.wv:
            raise KeyError(f"Il token '{token}' non esiste nel modello Word2Vec")
    # Ritorna il vettore di embedding associato al token
        return model.wv[token]

    def combine_vectors (input_vectors):
        return sum(input_vectors)/input_vectors.length()


    # Definizione del metodo predict, che restituisce la previsione della rete neurale
    def predict(self,input_phrase):
        input_tokens = tokenize(input_phrase) 
        input_vectors = [word2veclookup(token) for token in input_tokens]

        input_vector = combine_vectors(input_vectors)

        # Calcolo dell'output del primo layer
        layer_1 = np.dot(input_vector, self.weights) + self.bias 
        # Applicazione della funzione di attivazione sigmoid all'output del primo layer
        self.weights
        layer_2 = np.dot(layer_1, self.weights_2) + self.bias_2
        layer_3 = self._sigmoid(layer_2)
        # La previsione è l'output del terzo layer
        prediction = layer_3
        return prediction

    # Definizione del metodo per il calcolo dei gradienti
    def _compute_gradients(self, input_vector, target):
    # Calcolo dell'output del primo layer
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        # Applicazione della funzione di attivazione sigmoid all'output del primo layer
        layer_2 = self._sigmoid(layer_1)
        # La previsione è l'output del secondo layer
        prediction = layer_2
        # Calcolo della derivata dell'errore rispetto alla previsione
        derror_dprediction = 2 * (prediction - target)
        # Calcolo della derivata dell'output del secondo layer rispetto all'output del primo layer
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        # Derivata del primo layer rispetto al bias
        dlayer1_dbias = 1
        # Derivata del primo layer rispetto ai pesi
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        # Calcolo della derivata dell'errore rispetto al bias
        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        # Calcolo della derivata dell'errore rispetto ai pesi
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )
        # Restituzione delle derivate
        return derror_dbias, derror_dweights

# Definizione del metodo per l'aggiornamento dei parametri
def _update_parameters(self, derror_dbias, derror_dweights):
    # Aggiornamento del bias
    self.bias = self.bias - (derror_dbias * self.learning_rate)
    # Aggiornamento dei pesi
    self.weights = self.weights - (
        derror_dweights * self.learning_rate
    )

def train(self, input_phrases, targets, iterations):
    cumulative_errors = []
    for current_iteration in range(iterations):
        # Selezione casuale di un batch di dati di addestramento
        batch_size = 32
        random_indices = np.random.permutation(len(input_phrases))
        random_indices = random_indices[:batch_size]
        batch_input_phrases = [input_phrases[index] for index in random_indices]
        batch_targets = [targets[index] for index in random_indices]

        # Calcolo delle derivate parziali dell'errore rispetto ai parametri del modello
        derror_dbias, derror_dweights = self._compute_gradients(
            batch_input_phrases, batch_targets
        )

        # Aggiornamento dei pesi e dei bias utilizzando le derivate calcolate
        self._update_parameters(derror_dbias, derror_dweights)

        # Ogni 100 iterazioni, calcola l'errore cumulativo su tutto il dataset
        if current_iteration % 100 == 0:
            cumulative_error = 0
            for data_instance_index in range(len(input_phrases)):
                data_phrase = input_phrases[data_instance_index]
                target = targets[data_instance_index]
                prediction = self.predict(data_phrase)
                error = np.square(prediction - target)
                cumulative_error += error
            cumulative_errors.append(cumulative_error)

    return cumulative_errors



