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
    
def chromosomeFitness(population):  

    for chromosome in population:
        train_in = trainingData[:,:-1]
        train_tgt = trainingData[:,57:58]
        testing_in = testData[:,:-1]
        testing_tgt = testData[:,-1]
        
        # print("Fitness Training data selected shape",np.shape(train_in))
        # print("Fitness Training data selected target shape",np.shape(train_tgt))
        # print('fitness test data',np.shape(testData))
        # print('fitness test data',np.shape(trainingData))
        redTraining_in = removecolumns(trainingData,chromosome)
        # train_in = train_in*chromosome
        print(np.shape(redTraining_in))
        
        net = mlp.mlp(redTraining_in,train_tgt,10,outtype = 'logistic')#different types of out puts: linear, logistic,softmax
        
        error = net.mlptrain(redTraining_in,train_tgt,0.1,101)
       
        errorEarlyStoppingError = net.earlystopping(redTraining_in,train_tgt,redTraining_in,train_tgt,10)
        
        percentageAccuracy = net.confmat(redTraining_in,train_tgt)
        # percentageAccuracy = net.confmat(testing_in,testing_tgt)    
        # results[idx,1] = percentageAccuracy
        
        