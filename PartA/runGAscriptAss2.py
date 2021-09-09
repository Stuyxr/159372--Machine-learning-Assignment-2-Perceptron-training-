#parameters for the ga function: (stringLength,fitnessFunction,nEpochs,populationSize=100,mutationProb=-1,crossover='un',nElite=4,tournament=True)# g = ga.ga(100,'fF.fourpeaks',1000,100,0.01,'sp',4,False)# g.runGA()# ##nEpochs = number generations.'''function uses the best chromosome returned from genetic algorithm to reduce the number of features and train the mlp with data containing reduced number of columns'''import timeimport PartA.getData as getDataimport PartA.mlp as mlpimport PartA.ga as gaimport numpy as np'''reduce features of data'''def removecolumns(data,chromosome):    i = 0    reducedData_in = np.array([])    for value in chromosome:  #for each value in the chromosome that is 1, the column of training data set is included in a new array        if value == 1:            if((np.shape(reducedData_in)[0]) == 0):                reducedData_in = data[:,i]            else:                reducedData_in = np.column_stack((reducedData_in,data[:,i]))                        i+=1    # print("removed colmn data", np.shape(reducedData_in))        return reducedData_in 
#--------------------------------------------------------------------main----------------------------------------'''Obtain data to use with fitness function to train mlp '''testData,trainingData,validation = getData.runGetData()train_in = trainingData[:,:-1]train_tgt = trainingData[:,57:58]testing_in = testData[:,:-1]testing_tgt = testData[:,57:58]validation_in = validation[:,:-1]validation_tgt = validation[:,57:58]timeStart = time.time()print(timeStart)# g = ga.ga(57,'cF.chromosomeFitness',5,5,0.8,'un',4,False)'''pass data to GA to be used by GA and fitness function'''g = ga.ga(train_in,train_tgt,validation_in,validation_tgt,57,'cF.chromosomeFitness',100,10,0.1,'un',4,True)print("running GA")# timeStart = time.time()# print(timeStart)'''Get the best scoring chromosome to train and test mlp'''chrome = g.runGA()print("Chromosome Returned from genetic algorithm",chrome)'''Remove columns as per the chomosome in data and train mlp'''timeStop = time.time()print(" =======================> time stop ", timeStop - timeStart)testing_in = removecolumns(testing_in,chrome)train_in  = removecolumns(train_in,chrome)validation_in = removecolumns(validation_in,chrome)net = mlp.mlp(train_in,train_tgt,10,outtype = 'linear')#different types of out puts: linear, logistic,softmaxerrorEarlyStoppingError = net.earlystopping(train_in,train_tgt,validation_in,validation_tgt,0.1,10)percentageAccuracy = net.confmat(testing_in,testing_tgt)