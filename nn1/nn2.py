import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        # Calcola il valore della prima layer
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        # Applica la funzione sigmoid alla prima layer per ottenere l'output
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        # Calcola il valore della prima layer
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        # Applica la funzione sigmoid alla prima layer per ottenere l'output
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        # Calcola la derivata parziale dell'errore rispetto alla predizione
        derror_dprediction = 2 * (prediction - target)
        # Calcola la derivata della sigmoid della prima layer
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        # Calcola la derivata parziale della prima layer rispetto al bias
        dlayer1_dbias = 1
        # Calcola la derivata parziale della prima layer rispetto ai pesi
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        # Calcola la derivata parziale dell'errore rispetto al bias
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        # Calcola la derivata parziale dell'errore rispetto ai pesi
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        # Aggiorna il bias
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        # Aggiorna i pesi
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)
            self._update_parameters(derror_dbias, derror_dweights)
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
