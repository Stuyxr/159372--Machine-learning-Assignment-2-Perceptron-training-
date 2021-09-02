'''
Created on 1/09/2021

@author: yolan
'''

import PartB.getData as getData
import numpy as np
import PartB.som as som
import pylab as pl
if __name__ == '__main__':
    pass

testData,trainingData,validation = getData.runGetData()

train_in = trainingData[:,:-1]
traint = trainingData[:,57:58]
testing_in = testData[:,:-1]
testingt = testData[:,57:58]
validation_in = validation[:,:-1]
validation_tgt = validation[:,57:58]

bestGenome =[0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]

def removecolumns(data):
    i = 0
    reducedData_in = np.array([])
    for value in bestGenome:  #for each value in the chromosome that is 1, the column of training data set is included in a new array
        if value == 1:
            if((np.shape(reducedData_in)[0]) == 0):
                reducedData_in = data[:,i]
            else:
                reducedData_in = np.column_stack((reducedData_in,data[:,i]))
                
        i+=1
    # print("removed colmn data", np.shape(reducedData_in))    
    return reducedData_in 

def reduceSetData(train_inFull,testing_inFull,validation_inFull):
    trainingDataReduced = removecolumns(train_inFull)
    print(np.shape(trainingDataReduced))
    testDataReduced = removecolumns(testing_inFull)
    print(np.shape(testDataReduced))
    validationReduced = removecolumns(validation_inFull)
    return trainingDataReduced,testDataReduced,validationReduced



def SomData(train_in,traint,testing_in,testingt,validation_in,validation_tgt):
    
    
    print("Test data ",np.shape(testData))
    print("Training data",np.shape(trainingData))
    print("Validation data", np.shape(validation))
    
    train_tgt = np.transpose(traint)
    testing_tgt = np.transpose(testingt)
    print("training target shape: ",train_tgt)
    print("train data shape",np.shape(train_in))
    
    net = som.som(10,10,train_in,usePCA=0)
    net.somtrain(train_in,400)
    
    best = np.zeros(np.shape(train_in)[0],dtype=int)
    for i in range(np.shape(train_in)[0]):
        best[i],activation = net.somfwd(train_in[i,:])
    
    pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    where = pl.where(train_tgt == 0)[1]
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
    where = pl.where(train_tgt == 1)[1]
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
    pl.axis([-0.1,1.1,-0.1,1.1])
    pl.axis('off')
    
    pl.figure(2)

    print("next round")
    
    best = np.zeros(np.shape(testing_in)[0],dtype=int)
    for i in range(np.shape(testing_in)[0]):
        best[i],activation = net.somfwd(testing_in[i,:])
    
    pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    where = pl.where(testing_tgt == 0)[1]
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
    where = pl.where(testing_tgt == 1)[1]
    pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
    pl.axis([-0.1,1.1,-0.1,1.1])
    pl.axis('off')
    pl.show()
    
# SomData(train_in,traint,testing_in,testingt,validation_in,validation_tgt)

trainingDataReduced,testDataReduced,validationReduced = reduceSetData(train_in,testing_in,validation_in)
SomData(trainingDataReduced,traint,testDataReduced,testingt,validationReduced,validation_tgt)
