import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        #inposto i pesi per far in modo che successivamente il prodotto non possa venire 0
        self.weights = np.array([np.random.randn(), np.random.randn()])

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
        self.bias = np.random.randn()
        #prendo la variabile da ai1 e la metto nella variabile locale
        self.learning_rate = learning_rate


#    e una funzione non lineare (e stabile su un valore poi incrementa o decrementa per un lesso di tempo ben definito e infine si stabilizza su un altro valore)
#    nello specifico la sigmoide passa da (-n a n) sull asse delle x mentre sull asse delle y passa da 0 a 1 
#    proprio per questo viene utilizzata nelle ai OGNI numero che entra diventera 1
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#    definisco l'algoritmo per la derivata della sigmoide originale in modo da trovare ogni valore di calcolo più precisamente 
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

#    creo l'algoritmo della predizione
    def predict(self, input_vector):
        #    creo i layer per fare le predizzioni, servono per far comunicare (le diverse fasi del processo) i bias 
        #    layer di input/hiden
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

# calcolo i gradienti quindi di base faccio le derivate e derivate di derivate 
    def _compute_gradients(self, input_vector, target):
#    "creo" un layer per connettere l'inizzio del codice alla fine
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

#    calcolo la derivata del errore dela derivata predizione
        derror_dprediction = 2 * (prediction - target)
#
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1

        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )

        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):

        cumulative_errors = []

        for current_iteration in range(iterations):

            # Pick a data instance at random

            random_data_index = np.random.randint(len(input_vectors))


            input_vector = input_vectors[random_data_index]

            target = targets[random_data_index]


            # Compute the gradients and update the weights

            derror_dbias, derror_dweights = self._compute_gradients(

                input_vector, target

            )


            self._update_parameters(derror_dbias, derror_dweights)


            # Measure the cumulative error for all the instances

            if current_iteration % 100 == 0:

                cumulative_error = 0

                # Loop through all the instances to measure the error

                for data_instance_index in range(len(input_vectors)):

                    data_point = input_vectors[data_instance_index]

                    target = targets[data_instance_index]


                    prediction = self.predict(data_point)

                    error = np.square(prediction - target)


                    cumulative_error = cumulative_error + error

                cumulative_errors.append(cumulative_error)


        return cumulative_errors
