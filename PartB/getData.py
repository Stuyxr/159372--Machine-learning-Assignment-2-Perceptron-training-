'''
Created on 22/08/2021

@author: yolan
'''
from _overlapped import NULL

import PartB.mlp as mlp
import numpy as np
import pylab as pl


fileName = "spambase.data"



def readFromFile(fileName):
    spamDataFromFile = np.loadtxt(fileName,delimiter=',')
    return spamDataFromFile
   

def dataPlot(spamData):
    spam = np.where(spamData[:,-1] ==1) 
    ham = np.where(spamData[:,-1] ==0)
    pl.plot(spamData[spam,1],'ro')
    pl.plot(spamData[ham,1],'go')
    
    pl.show()
    
# normalize numerical data 
def normalizeData(newArrayData,column):
    minValue = np.min( newArrayData[:,column])
    maxValue = np.max( newArrayData[:,column])
    
    average = (maxValue -minValue)
    if(average == 0):
        return newArrayData
    else:
        newArrayData[:,column] = newArrayData[:,column]-minValue
        newArrayData[:,column] = (newArrayData[:,column])/average
        # print(newArrayData[:,column])
        # print(np.min(newArrayData[:,column]))
        return newArrayData
    
def ShuffleDataRandomly(newArrayData):
    # target = newArrayData[:,-1]
    order = np.arange(np.shape(newArrayData)[0])
    np.random.shuffle(order)
    newArrayData = newArrayData[order,:]
    return newArrayData

# def getRandomRow(DataArray):
#     DataArray = ShuffleDataRandomly(DataArray)
#     row = DataArray[:1]
#     return row

def AddtoArray(NewArrayData,row,row_n):
    if np.shape(NewArrayData)[0]== 0:
                NewArrayData = row
    else:
        NewArrayData = np.append(NewArrayData,row,axis=0)
    return NewArrayData

def deleteRow(df,row):
    newData = np.delete(df,row, axis=0)
    return newData

    # create a data set with a 1:1 ratio of yes and no values
def BalanceSampling(DataArray, sizeArrayData):
    yesCounter = 0 # count the number yes target values is in the newArrayData
    noCounter = 0 #count the number no target values is in the newArrayData
    counter = 1 # make sure the max amount of data rows is not exceeded  
    DataArray = ShuffleDataRandomly(DataArray)
    numberYes = round(sizeArrayData *0.5) #Divide the data 50% maximum number yes values to be added to the array
    numberNo = round(sizeArrayData *0.5) #Divide the data 50% no
    # print("number Yes values",np.shape(np.where(DataArray[:,-1] == 1)))
    # print("number No values", np.shape(np.where(DataArray[:,-1] == 0)))
 

    
    Test =[]
    Train =[]

    while(counter <= np.shape(DataArray)[0]): 
        row = DataArray[counter-1:counter] # get the first row of randomly shuffled array
        valueYesOrNo = row[:,57:58] #get the last column containing output yes or no data  
        if valueYesOrNo == 1 and yesCounter < numberYes:
            Test = AddtoArray(Test,row,counter)
            # DataArray = deleteRow(DataArray,0) # delete row from data set so that its not selected twice
            yesCounter+=1
            # print("Yes: ", yesCounter, " Yes Counter: ", numberYes, "Counter", counter, "Special: ", np.shape(DataArray)[0], "Shape: ", np.shape(Test))
          
        elif valueYesOrNo == 0 and noCounter < numberNo:
            Test = AddtoArray(Test,row,counter)    
            # DataArray = deleteRow(DataArray,0)
            noCounter+=1
            # print("No: ", noCounter, " No Counter: ", numberNo, "Counter", counter, "Special: ", np.shape(DataArray)[0], "Shape: ", np.shape(Test))
        else:  
             Train = AddtoArray(Train,row,counter)
        
        counter+=1
        # print(DataArray[:1])
    # print("Done - Counter", counter, "Special: ", np.shape(DataArray)[0], "Shape: ", np.shape(Test))
    print(np.shape(Test))
    return  Test, Train

def seperateData(df,percentageTesting):
    percentageTesting = int(percentageTesting)
    print(percentageTesting)
    data = ShuffleDataRandomly(df)
    BalancedTestingdata = data[:percentageTesting,:]
    BalanceTrainingData = data[percentageTesting+1:,:]
    
    return BalancedTestingdata, BalanceTrainingData 
    
    # Separate training 70% from testing data  30%
def seperateData70vs30(df,percentageTesting):
    testData, trainingData = BalanceSampling(df,percentageTesting)
    return testData, trainingData 
    
#-------------------------------------------------------main----------------------------------------------
'''
this function obtain all the data from the data file.
Data is balanced, Normalized using max-min normalization and seperated into training,testing and validation data
'''
def runGetData():
    print("Read data from file")
    spamData = readFromFile(fileName)
    newArray = np.array([])
    
    # print("Datafrom file",spamData[:10])

    # nullData = np.where(spamData[:] == None)
    # print(nullData)
    # dataPlot(spamData)
    
    spamData,validData = BalanceSampling(spamData,1002)
    
    print("number Yes values",np.shape(np.where(spamData[:,-1] == 1)))
    print("number No values", np.shape(np.where(spamData[:,-1] == 0)))
    
    # print(np.shape(spamData))
    
    '''
    Normalizing data
    '''
    # print("Normalizing data max-min")
    for column in range(np.shape(spamData)[1]):
         spamData = normalizeData(spamData,column)
    


    # print("number Yes values",np.shape(np.where(spamData[:,-1] == 1)))
    # print("number No values", np.shape(np.where(spamData[:,-1] == 0)))
    
    '''
    Seperate data into test and training set. 
    '''
    print("Separate data  30% testing 70% training")
    sizeTestData = round(((np.shape(spamData)[0])*0.3),0)
       
    testData, trainingData =seperateData(spamData,sizeTestData)
    testData = ShuffleDataRandomly(testData)
    trainingData = ShuffleDataRandomly(trainingData)
    
    SizeValidationData = round(((np.shape(trainingData)[0])*0.25),0)
    validation,trainingData = seperateData(trainingData,SizeValidationData)
    validation = ShuffleDataRandomly(validation)
    # print("Test Data shape",np.shape(testData))
    # print("Training Data shape",np.shape(trainingData))
    # print("Training data",trainingData[:10])
    # print("Testing data",testData[:10])
    
    # print("number Yes values test data",np.shape(np.where(testData[:,-1] == 1)))
    # print("number No values test Data", np.shape(np.where(testData[:,-1] == 0)))
    # print("number Yes values training data",np.shape(np.where(trainingData[:,-1] == 1)))
    # print("number No values training Data", np.shape(np.where(trainingData[:,-1] == 0)))
    

    return testData,trainingData,validation
'''
function used to test output of MLP without GA 
'''
def runMLP(trainingData,testData,validation):
    
    train_in = trainingData[:,:-1]
    train_tgt = trainingData[:,57:58]
    testing_in = testData[:,:-1]
    testing_tgt = testData[:,57:58]
    validation_in = validation[:,:-1]
    validation_tgt = validation[:,57:58]
    
    # print("Training data", train_in[:10])
    # print("Testing data", testing_in[:10])
    
    # for i in [5,10,15,20,25,30,35,40,45,50]:
    for i in [2,5,10,15,20,25]:
    
        net = mlp.mlp(train_in,train_tgt,i,outtype = 'linear')
        
        # net = mlp(train_in,traint_gt,i,outtype = 'linear')#different types of out puts: linear, logistic,softmax
        # error = net.mlptrain(train_in,train_tgt,0.1,5001)
        errorEarlyStoppingError = net.earlystopping(train_in,train_tgt,validation_in,validation_tgt,0.1,5000)
        percentageAccuracy = net.confmat(testing_in,testing_tgt)   
        # percentageAccuracy = net.confmat(testing_in,testing_tgt)    

        
# testMLP()
# testData,trainingData,validation = runGetData()
# runMLP(trainingData,testData,validation)