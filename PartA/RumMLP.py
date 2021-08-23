'''
Created on 22/08/2021

@author: yolan
'''
import numpy as np
from _overlapped import NULL
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
def normalizeData2(newArrayData,column):
    minValue = np.min( newArrayData[:,column])
    maxValue = np.max( newArrayData[:,column])
    
    average = (maxValue -minValue)
    if(average == 0):
        return newArrayData
    else:
        newArrayData[:,column] = newArrayData[:,column]-minValue
        newArrayData[:,column] = newArrayData[:,column]/average
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
    counter = 0 # make sure the max amount of data rows is not exceeded  
    DataArray = ShuffleDataRandomly(DataArray)
    numberYes = round(sizeArrayData *0.5) #Divide the data 50% maximum number yes values to be added to the array
    numberNo = round(sizeArrayData *0.5) #Divide the data 50% no
    
    NewArrayData =[]


    while(counter <= np.shape(DataArray)[0]): 
        row = DataArray[:1]
        # print(DataArray[:1])
        valueYesOrNo = row[:,-1]  
        if valueYesOrNo == 1 and yesCounter <= numberYes:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            yesCounter+=1
    
          
        if valueYesOrNo == 0 and noCounter < numberNo:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            noCounter+=1
         
        counter+=1
        # print(DataArray[:1])
    return  NewArrayData, DataArray
    
    # Separate training 70% from testing data  30%
def seperateData70vs30(df,percentageTesting):
    testData, trainingData = BalanceSampling(df,percentageTesting)
    return testData, trainingData 
    
#-------------------------------------------------------main----------------------------------------------

spamData = readFromFile(fileName)
print(np.shape(spamData))
nullData = np.where(spamData[:] == None)
print(nullData)
# dataPlot(spamData)

newspamData,validData = BalanceSampling(spamData,3626)

print(np.shape(newspamData))

'''
Seperate data into test and training set. 
'''
sizeTestData = (np.shape(newspamData)[0])*0.3

testData, trainingData =seperateData70vs30(newspamData,sizeTestData)



