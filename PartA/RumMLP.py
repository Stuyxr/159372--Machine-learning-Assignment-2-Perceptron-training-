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
   

def dataPlot():
    pl.plot(spamData[Spam,1],'ro')
    pl.plot(spamData[NotSpam,1],'go')
    
    pl.show()
    
#-------------------------------------------------------main----------------------------------------------

spamData = readFromFile(fileName)
print(np.shape(spamData))
nullData = np.where(spamData[:] == None)
print(nullData)

Spam = np.where(spamData[:,-1] ==1) 

NotSpam = np.where(spamData[:,-1] ==0) 

pl.plot(spamData[Spam,0],spamData[Spam,1],'ro')
pl.plot(spamData[NotSpam,0],spamData[NotSpam,1],'go')
pl.show()
