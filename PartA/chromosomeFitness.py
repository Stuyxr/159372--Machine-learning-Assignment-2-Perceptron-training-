'''
Created on 22/08/2021

@author: yolan
'''
import getData
import numpy as np
import mlp
from scipy.special import expit


testData,trainingData = getData.runGetData()

def removecolumns(trainingData,chromosome):
    i = 0
    reducedTraining_in = np.array([])
    for value in chromosome:
        if value == 1:
            if((np.shape(reducedTraining_in)[0]) == 0):
                reducedTraining_in = trainingData[:,i]
            else:
                reducedTraining_in = np.column_stack((reducedTraining_in,trainingData[:,i]))
                
        i+=1
    # print("removed colmn data", np.shape(reducedTraining_in))    
    return reducedTraining_in        
    
def chromosomeFitness(pop): 
    percentageAccuracy = 0
    fitness = np.zeros(np.shape(pop)[0]) 
    index =0
    for chromosome in pop:
        train_in = trainingData[:,:-1]
        train_tgt = trainingData[:,57:58]
        testing_in = testData[:,:-1]
        testing_tgt = testData[:,-1]
        
        # print("Fitness Training data selected shape",np.shape(train_in))
        # print("Fitness Training data selected target shape",np.shape(train_tgt))
        # print('fitness test data',np.shape(testData))
        # print('fitness test data',np.shape(trainingData))
        train_in = removecolumns(trainingData,chromosome) # array after removing columns according to chromosome
        # train_in = train_in*chromosome
        print(np.shape(train_in))
        
        net = mlp.mlp(train_in,train_tgt,10,outtype = 'logistic')#different types of out puts: linear, logistic,softmax
        
        error = net.mlptrain(train_in,train_tgt,0.1,101)
       
        # errorEarlyStoppingError = net.earlystopping(train_in,train_tgt,train_in,train_tgt,10)
        
        percentageAccuracy = net.confmat(train_in,train_tgt)
        print(np.dtype(percentageAccuracy))
        # percentageAccuracy = net.confmat(testing_in,testing_tgt) 
        numberColumns =  np.shape(train_in)[1]
        overAllScore = percentageAccuracy + ((57-numberColumns)/57)*100  # get the maximum score add percentage accuracy to the fraction of max amount of columns minus the number columns per genome
         
        fitness[index] = overAllScore
        index +=1
    return fitness
        