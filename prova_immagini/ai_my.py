
from PIL import Image
import numpy as np
import nn2

input_vector = np.array(Image.open('Z:/tutto/python/ai3/cumulative_error.png'))

#creo i target
for x in range(len(input_vector)):
    x = x
x+=1
targets = np.random.randint(2,size=x)
learning_rate = 0.01
neural_network = nn2.NeuralNetwork(learning_rate)
a=neural_network.predict(input_vector)
print(int(a[0][0]))

image = Image.fromarray(a)
image.save('prova_immagini/c_err_prova.png')

# (480, 640, 4)

# im_R = im.cop\y()
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



