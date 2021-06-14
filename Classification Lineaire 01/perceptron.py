#BELVAL GIOVANNI

#ecriture de la classe du perceptron.

#import des modules intéressant : 

import numpy as np
import time

#! fonction de discriminination prevue a l'avance : 

def model(x):
    return -1 if x[0] <= 0 else 1



class Perceptron(object):

    #constructeur du perceptron : 

    def __init__(self,LERANING_RATE = 0.1 , data = "data.txt"):
        #poids Synaptique Incluant le bias.
        self.weights = np.random.random(3)
        self.learning_rate = LERANING_RATE
        self.data = np.loadtxt(data)

    
    def guess(self,x):
        # utilisatipn de l fonction sgn comme seuil d'activation du perceptron : 
        x = np.hstack(([1],x))

        return 1 if self.weights.T @ x > 0 else -1

    def Up(self , n = 100):
        #* fonction d'amelioration du perceptron :
        
        for i in range(n):
            for i in range(self.data.shape[0]):
                x = np.hstack(([1] , self.data[i,:]))
            
                for j in range(self.weights.size):
                    self.weights[j] += self.learning_rate * (model(self.data[i,:]) - self.guess(self.data[i,:])) * x[j]


        return


    def get_weights(self):
        return self.weights.copy()

        




