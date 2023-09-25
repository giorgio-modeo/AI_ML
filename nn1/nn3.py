import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec



model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


# Definizione della classe NeuralNetwork
class NeuralNetwork:
    # Definizione del metodo init, che inizializza i pesi, il bias e il learning rate
    def __init__(self, learning_rate):
        # Inizializzazione dei pesi con valori casuali
        self.weights = Word2Vec.wv.vectors
        self.weights_2 = Word2Vec.wv.vectors

        self.bias = np.random.normal(loc=0.0,scale=0.01,size=20)
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
    
    def word2vec_lookup(model, token):
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
        input_vectors = [word2vec_lookup(token) for token in input_tokens]

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

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # seleziona casualmente un dato di input e il suo obiettivo corrispondente
            # La linea seguente estrae casualmente un indice di input_vectors e targets
            # su cui effettuare la successiva iterazione del training
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Calcola le derivate parziali dell'errore rispetto ai parametri del modello
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            # Aggiorna i pesi e il bias utilizzando le derivate calcolate in precedenza
            self._update_parameters(derror_dbias, derror_dweights)

            # Ogni 100 iterazioni, calcola l'errore cumulativo su tutto il dataset di input_vectors
            if current_iteration % 100 == 0:
                cumulative_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)
                    cumulative_error += error

                cumulative_errors.append(cumulative_error)


        return cumulative_errors
