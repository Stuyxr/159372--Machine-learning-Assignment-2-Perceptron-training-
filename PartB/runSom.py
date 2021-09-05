'''
Created on 1/09/2021

@author: yolan
'''

import PartB.getData as getData
import numpy as np
import PartB.som as som
import pylab as pl
import PartB.pcn as pcn
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
'''
Remove columns of data set to reduce features according to the best chromosome in part A
'''
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

def determineBestNode():
    pass

def countoverlaps(traint,best):
    train_tgt = np.transpose(traint)
     # Find places where the same neuron represents different classes
    i0 = np.where(train_tgt==0)[1]
    nodes0 = np.unique(best[i0])
    i1 = np.where(train_tgt==1)[1]
    nodes1 = np.unique(best[i1])
    
    doubles01 = np.in1d(nodes0,nodes1,assume_unique=True)
    
    return  len(nodes0[doubles01]) 
   


def SomData(x,y,train_in,traint,testing_in,testingt,validation_in,validation_tgt):
    
    train_tgt = np.transpose(traint)
    testing_tgt = np.transpose(testingt)
    
    net = som.som(x,y,train_in,usePCA=1)
    net.somtrain(train_in,100)
    
    # Store the best node for each training input
    bestTrain = np.zeros(np.shape(train_in)[0],dtype=int)
    bestAtivation = np.zeros(np.shape(train_in)[0],dtype=float)
    for i in range(np.shape(train_in)[0]):
        bestTrain[i],activationTrain = net.somfwd(train_in[i,:])
        bestAtivation[i] = activationTrain[bestTrain[i]]
        
    # print(bestTrain)
    print("best shape 1 ", np.shape(bestTrain))
    # print("activation Training", activationTrain)
    pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    where = pl.where(train_tgt == 0)[1]
    pl.plot(net.map[0,bestTrain[where]],net.map[1,bestTrain[where]],'rs',ms=30)
    where = pl.where(train_tgt == 1)[1]
    pl.plot(net.map[0,bestTrain[where]],net.map[1,bestTrain[where]],'gv',ms=30)
    pl.axis([-0.1,1.1,-0.1,1.1])
    pl.axis('off')
    
    # pl.figure(2)

    # print("next round")
    #
    #  # Store the best node for each training input
    # bestTest = np.zeros(np.shape(testing_in)[0],dtype=int)
    # for i in range(np.shape(testing_in)[0]):
    #     bestTest[i],activationTest = net.somfwd(testing_in[i,:])
    #
    # print(bestTest)
    # print("best shape 2 ", np.shape(bestTest))
    # # print("activation testing",activationTest)
    # pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
    # where = pl.where(testing_tgt == 0)[1]
    # pl.plot(net.map[0,bestTest[where]],net.map[1,bestTest[where]],'rs',ms=30)
    # where = pl.where(testing_tgt == 1)[1]
    # pl.plot(net.map[0,bestTest[where]],net.map[1,bestTest[where]],'gv',ms=30)
    # pl.axis([-0.1,1.1,-0.1,1.1])
    # pl.axis('off')
    pl.show()
    # return(activationTrain,activationTest,bestTrain,bestTest)
    return(bestTrain,bestAtivation)
    
'''
Using full data set uncomment  below determine the best number of nodes

'''
# score = np.zeros((225,1))
# count = 0
# lowestScore = 1000
# lowestX = 0
# lowestY = 0
# for x in [15]:
# # for x in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]:
# #     for y in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]:
#     for y in [15]:
#         best,activation =SomData(x,y,train_in,traint,testing_in,testingt,validation_in,validation_tgt)
#         NumberOverlapse = countoverlaps(traint,best)
#         score[count] = (NumberOverlapse/(x*y))*100
#         if score[count] < lowestScore:
#             lowestScore = score[count]
#             lowestX = x
#             lowestY = y
#         count +=1
# print(score)
# print("lowest score x",lowestX)
# print("Lowest score y",lowestY)
# print("Lowest score", lowestScore)
# # Pick the best
# print("Best low score ====================>", np.argmin(score))


'''
Using reduced feature data set to run the SOM on the chosen numbers of nodes for the map

'''

# trainingDataReduced,testDataReduced,validationReduced = reduceSetData(train_in,testing_in,validation_in)
# # activationTrain,activationTest,bestTrain,bestTest = SomData(2,2,trainingDataReduced,traint,testDataReduced,testingt,validationReduced,validation_tgt)
# bestTrain,activation = SomData(15,10,trainingDataReduced,traint,testDataReduced,testingt,validationReduced,validation_tgt)

'''
using SOM activation as input to perceptron  
'''
best,activation =SomData(20,20,train_in,traint,testing_in,testingt,validation_in,validation_tgt)
print("Activation Training set ", np.shape(activation))

# print("Activation Testing set ", np.shape(activationTest))
# print("Activation Testing set ", activationTest)
# print("training", np.shape(train_in), np.shape(traint))
# print("testing", np.shape(testing_in), np.shape(testingt))
activationTrans = activation.reshape(-1,1)
p = pcn.pcn(train_in,traint)
p.pcntrain(activationTrans,train_in,traint,0.1,1000)
p.confmat(train_in,traint)
print("------------------------------------------------Finished--------------------------------------------------")
