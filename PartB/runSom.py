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
print("Test data ",np.shape(testData))
print("Training data",np.shape(trainingData))
print("Validation data", np.shape(validation))

train_in = trainingData[:,:-1]
train_tgt = trainingData[:,57:58]
testing_in = testData[:,:-1]
testing_tgt = testData[:,57:58]
validation_in = validation[:,:-1]
validation_tgt = validation[:,57:58]



net = som.som(6,6,train_in,usePCA=0)
net.somtrain(train_in,400)

best = np.zeros(np.shape(train_in)[0],dtype=int)
for i in range(np.shape(train_in)[0]):
    best[i],activation = net.somfwd(train_in[i,:])

print(best)
print(np.shape(best))
pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.where(train_tgt == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
where = pl.where(train_tgt == 1)

pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.figure(2)