'''
Created on 22/08/2021

@author: yolan
'''
# import getData
from scipy.special import expit

import numpy as np


import PartA.mlp as mlp

higestScoring = 0.0
higestScoringChomo = ([])



def removecolumns(trainingData,chromosome):
    i = 0
    reducedTraining_in = np.array([])
    for value in chromosome:  #for each value in the chromosome that is 1, the column of training data set is included in a new array
        if value == 1:
            if((np.shape(reducedTraining_in)[0]) == 0):
                reducedTraining_in = trainingData[:,i]
            else:
                reducedTraining_in = np.column_stack((reducedTraining_in,trainingData[:,i]))
                
        i+=1
    # print("removed colmn data", np.shape(reducedTraining_in))    
    return reducedTraining_in        

       
    
'''
The function receive data sets from GA that is used to determine the scores of the chromosomes. 
scores are determined by obtaining the cross validation accuracy and adding 
fraction of max amount of columns minus the number columns per genome. 
'''    
def chromosomeFitness(pop,trainingData,train_tgt,validation,validation_tgt): 
    percentageAccuracy = 0
    fitness = np.zeros(np.shape(pop)[0]) 
    index =0
    for chromosome in pop:
    
        train_in = removecolumns(trainingData,chromosome) # array after removing columns according to chromosome
        validation_in = removecolumns(validation,chromosome)
        
        net = mlp.mlp(train_in,train_tgt,10,outtype = 'linear')#different types of out puts: linear, logistic,softmax
       
        errorEarlyStoppingError = net.earlystopping(train_in,train_tgt,validation_in,validation_tgt,0.1,10)
        
        percentageAccuracy = net.confmat(validation_in,validation_tgt)
        # percentageAccuracy = net.confmat(testing_in,testing_tgt) 
        numberColumns =  np.shape(train_in)[1]
        overAllScore = percentageAccuracy + ((57-numberColumns)/57)*100  # get the maximum score add percentage accuracy to the fraction of max amount of columns minus the number columns per genome
        fitness[index] = overAllScore

        index +=1
    return fitness
        