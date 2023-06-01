import os

def cls():
    os.system("cls")

def controllo_int(s):
    while True:
        try:
            n = int(input(s))
            break
        except:
            print("Error")
    return n

def controllo_float(s):
    while True:
        try:
            f = float(input(s))
            break
        except:
            print("Error")
    return f

def controllo_string(s):
    stringa = input(s)
    while True:
        if stringa == (""or" "):
            print("input invalido")
        else:
            return stringa

#                                                            esempio utilizzo di controllo stringa
# a = controllo_string("inserisci una parola")
# print(a)

def min_numero(num,minimo):
    if num > minimo:
        return num
    else:
        print("il numero e troppo piccolo")

def max_numero(num,massimo):
    if num < massimo:
        return num
    else:
        print("il numero e troppo grande")

def min_max_numero(num,minimo,massimo):
    if num < minimo and num > massimo:
        return num
    else:
        if num > massimo:
            return f'il numero "{num}" e troppo grande'
        else:
            return f'il numero "{num}" e troppo piccolo'
        

# inizzializzo un array con tutte le voci del menu il primo e il nome del menu
# uscita lo mette in automatico 
def menu(titolo,voci:list):
    menu = "*"*42 +"\n"
    menu += f"* {titolo}\n"
    menu += "*"*42 +"\n"
    for i in range(len(voci)):
        menu += "* "+str(i+1) + ". " + f"{voci[i]}" + "\n"
    menu += f"*\n* "+str(i+2) + f". uscita\n"
    menu += "*"*42
    return menu
#                                                            esempio utilizzo del menu
# voci = ["opzione 1","opzione 2","opzione 3","opzione 4"]
# print(menu("titolo",["opzione 1","opzione 2","opzione 3","opzione 4"]))

