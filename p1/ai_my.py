# import numpy as np
# saluti = ['ciao','salve','buongiorno','buonasera']
# nrandom = np.random.randint(len(saluti))


# richiesta=input("richiesta: ")
# richiestal=richiesta.split()
# nstringa=int("C")
# print(nstringa)
# for i in saluti:
#     for j in richiestal:
#         if i == j:
#             print(i)

# text = input("enter a string to convert into ascii values:")
# splt = text.split()
# ascii_values = []
# i =0

# print()
# for character in text:
#     ascii_values.append(ord(splt[i]))
#     i+=1
# print(ascii_values)


from PIL import Image
import numpy as np
import nn2

input_vector = np.array(Image.open('Z:/tutto/python/ai3/cumulative_error.png'))

print(input_vector)

#creo i target
for x in range(len(input_vector)):
    x = x
    pass
targets = np.random.randint(2,size=x)

learning_rate = 0.01
neural_network = nn2.NeuralNetwork(learning_rate)
neural_network.predict(input_vector)
training_error = neural_network.train(input_vector, targets, 10000)

# (480, 640, 4)

# im_R = im.copy()
# im_R[:, :, (1, 2)] = 0
# im_G = im.copy()
# im_G[:, :, (0, 2)] = 0
# im_B = im.copy()
# im_B[:, :, (0, 1)] = 0

# im_RGB = np.concatenate((im_R, im_G, im_B), axis=1)
# # im_RGB = np.hstack((im_R, im_G, im_B))
# # im_RGB = np.c_['1', im_R, im_G, im_B]
# image= Image.fromarray(im)
# image.save('p1/cumulative_error_copy.png')

# pil_img_rgb = Image.fromarray(im_RGB)
# pil_img_rgb.save('p1/cumulative_error_split.png')



