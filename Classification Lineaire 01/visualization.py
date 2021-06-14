#BELVAL GIOVANNI


import numpy as np
import matplotlib.pyplot as plt
import perceptron as p


#import du dataset : 

dataset = np.loadtxt("data.txt")
print(dataset)

# droite de séparation verticale : 

d = np.array([0,0])
d_ = np.array([-100,100])


#affichage des données du dataset : 

plt.scatter(dataset[:,0],dataset[:,1],label = "dataset")

#traitement du Dataset par le perceptron :
perceptron = p.Perceptron()
perceptron.Up()  #! Algorithme d'apprentissage du perceptron

#test sur un des points du dataset
print(f" nous testons le point {dataset[1,:]} et obtenons une prediction : {perceptron.guess(dataset[1,:])}")
print(f" le modele de prediction nous donne une valeur exate pour ce point de : {p.model(dataset[1,:])} ")


n = 0
for i in range(dataset.shape[0]):
    print(f" nous testons le point {dataset[i,:]} et obtenons une prediction : {perceptron.guess(dataset[i,:])}")
    print(f" le modele de prediction nous donne une valeur exate pour ce point de : {p.model(dataset[i,:])} ")

    #nous alons faire un ratio pour determiné la performance sur l'exemple d'entrainement : 
    if perceptron.guess(dataset[i,:]) == p.model(dataset[i,:]):
        n += 1

print(f"la performance sur le dataset de notre modele est de : {(n/dataset.shape[0])*100} %")
print(f"nous obtenons finalement : w = {perceptron.get_weights()}")


# dans la suite nous allons représenté graphiquement le model initial ainsi que notre séparation linéaire

# creation de la droite de séparation prédite : 
w = perceptron.get_weights()
s_ = np.linspace(-2,2,1000)
sep_prediction = -(w[1]/w[2]) * s_ - (w[0]/w[2])
plt.plot(s_,sep_prediction,label = "prediction",c= "r")

plt.plot(d,d_,label = "original")
plt.xlabel("axe des x1")
plt.ylabel("axe des x2")
plt.legend()
plt.show()