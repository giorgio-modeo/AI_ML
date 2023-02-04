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

text = input("enter a string to convert into ascii values:")
splt = text.split()
ascii_values = []
i =0

print()
for character in text:
    ascii_values.append(ord(splt[i]))
    i+=1
print(ascii_values)
