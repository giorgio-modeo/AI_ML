#inteligenza artificiale
#importo numpy per i calcoli complessi
import numpy as np
import random as r
#questo segmento e per fare un esempio 

#inmetto i punti dei vettori
input_vector = [1.72, 1.23]

#inposto i pesi per far in modo che successivamente il prodotto non possa venire 0
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# calcolo il prodotto 
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult
dot_product_1 = np.dot(input_vector, weights_1)
dot_product_2 = np.dot(input_vector, weights_2)

#stamp di verifica
print(f"The dot product is: {dot_product_1}")
print(f"The dot product is: {dot_product_2}")

