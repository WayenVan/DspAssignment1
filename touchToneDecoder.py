#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:15:25 2020

@author: wayenvan
"""
"""import essential modules"""
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

"""define globael number"""
dtmfHighFrequency = (1209, 1336, 1477, 1633)
dtmfLowFrequency = (697, 770, 852, 941)
dtmfLetter = [['1', '2', '3', 'A'],
              ['4', '5', '6', 'B'],
              ['7', '8', '9', 'C'],
              ['*', '0', '#', 'D']]

class ToneFlag(Enum):
    low = 0
    high = 1
    NoFind = 2

"""define functions"""
def generateXf(sampleRate, N):
    """generateXf for frequeny domain"""
    return np.linspace(0.0, (N-1)*sampleRate/N, N)

def generateXt(sampleRate, N):
    """generateXt for time domain"""
    return np.linspace(0.0, (N-1)*1/sampleRate, N)
    
def mag2dB(yf):
    """ change magnitude into dB form """
    return 20*np.log10(yf)

def peakFinding(data):
    """finding the max value of an array"""
    maxIndex = -1
    maxValue = 0
    
    for i in range(len(data)):
        if(data[i]>maxValue):
            maxIndex = i
            maxValue = data[i]
        
    return maxIndex

def peakFindingDouble(data):
    """finding the first 2 greatest value of an array"""
    indexMax = peakFinding(data[1:])+1
    
    indexTemp1 = peakFinding(data[1:indexMax-1])+1
    indexTemp2 = peakFinding(data[indexMax+1:])+indexMax+1
    
    if(data[indexTemp1]>=data[indexTemp2]):
        indexMaxSec = indexTemp1
    elif(data[indexTemp2]>data[indexTemp1]):
        indexMaxSec = indexTemp2
        
    ret = [indexMaxSec, indexMax]
    
    return ret

def aliasingFrequency(fs, sampleRate):
    """convert the signal frequency into (0, N/2)"""
    N = int(fs/sampleRate+0.5)
    return abs(fs-N*sampleRate)

def findFrequencyBelong(f, dtmfMin, dtmfMax, sampleRate):
    """
    Parameters
    ---------
    f: 
        input frequency of the chunk signal
    dtmfMin: 
        acceptable lower bound 
    dtmfMax: 
        acceptable high bound
    sampleRate: 
        sampling frequency
    
    Returns
    ------
    flag: 
        define which tone of the frequency belongs (high or low) 
    index: 
        return the index of dtmfFrequency array for easy finding of letter
    
    """

    for indexLow in range(4):
        for indexHigh in range(4):
            if(f-dtmfMin<aliasingFrequency(dtmfHighFrequency[indexHigh], sampleRate)<f+dtmfMax):
                flag = ToneFlag.high
                return flag, indexHigh
            if(f-dtmfMin<aliasingFrequency(dtmfLowFrequency[indexLow], sampleRate)<f+dtmfMax):
                flag = ToneFlag.low
                return flag, indexLow
    return ToneFlag.NoFind, -1
        
def detectOneDigitFromChunk(data, sampleRate):
    """
    detect each chunk
    
    Parameters
    ----------
    data: ndarray
        series of chunk data in time domain
    sampleRate: float
        the sampling frequency of signal

    Returns
    -------
    letter: string
        goal letter of this chunk 'N' means no letter found
    """
    #prepare the data
    dataf = np.fft.fft(data) 
    N=len(data)
    minMagnitude = 30
    #cut the data half
    rdataf = dataf[0:N//2]
    
    dtmfMin = 9
    dtmfMax = 9
    
    #calculate the peak point
    #ind = np.argpartition(abs(rdataf), -3)[-3:]
    ind = peakFindingDouble(abs(rdataf))
    
    #cut out small signal
    if((2/N*(abs(rdataf)[ind[0]])<minMagnitude) | (2/N*(abs(rdataf)[ind[1]])<minMagnitude)):
        return 'N'
    
    f1 = ind[0]*(sampleRate/N)
    f2 = ind[1]*(sampleRate/N)
    
    #print(f1, f2)
    #start the for loop to check if the frequency meet any of high or low frequency of dtmf
    (flag1, index1) = findFrequencyBelong(f1, dtmfMin, dtmfMax, sampleRate)
    (flag2, index2) = findFrequencyBelong(f2, dtmfMin, dtmfMax, sampleRate)
    
    #find out corresponding point of this 2 frequency
    if((flag1==ToneFlag.high) & (flag2==ToneFlag.low)):
        return dtmfLetter[index2][index1]
    elif((flag1==ToneFlag.low) & (flag2==ToneFlag.high)):
        return dtmfLetter[index1][index2]
    elif((flag1==ToneFlag.NoFind)|(flag2==ToneFlag.NoFind)):
        #print("index1:", index1, flag1, "index2", index2, flag2)
        return 'N'
    else:
        return 'N'

def autoDetectNumbers(data, sampleRate):
    """
    
    Parameters
    ----------
    data : ndarray
        touch tone data
    sampleRate : int or float
        sampling frequency

    Returns
    -------
    seriesNumber : String
       the number detected

    """
    K = 0
    N = len(data)
    gap = 300          #the length of eah chunk
    T = 1/sampleRate
    
    preResult = 'N'
    seriesNumber = ''
    
    #start checking numbers
    print("start finding raising edge chunk")
    while gap-1+K*gap < N:
        result = detectOneDigitFromChunk(data[K*gap: gap-1+K*gap], sampleRate)
        if((preResult=='N') & (result != 'N')):
            print(K*gap*T,'s', "-", (gap-1+K*gap)*T,'s:',result)
            seriesNumber = seriesNumber + result
        preResult = result
        K = K + 1
        
    return seriesNumber

"""main function"""
#load .dat file, if change i, it can load all files
figurePath = "./Output/Figures/"

dataAddress = 'touchToneData.dat'
dataI = np.loadtxt(dataAddress, usecols=(1), dtype=np.int16)
    
data = dataI
Fs2 = 1000 
N2 = len(data)
x2 = range(N2)
xt2 = generateXt(Fs2, N2)
xf2 = generateXf(Fs2, N2)
dataf = np.fft.fft(data)
    
series = autoDetectNumbers(data, Fs2)
print(dataAddress+":")
print("final result: ", series)

"""plot and save all figures"""
#plot task5 wave
# plt.figure(figsize=(20,10))
# plt.plot(xt2, data)
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.savefig(figurePath+"DTMFtime.pdf")

# plt.figure(figsize=(20,10))
# plt.plot(xf2[0:N2//2], mag2dB(abs(2/N2*dataf[0:N2//2])))
# plt.title("Frequency domain")
# plt.xlabel("Freqency(Hz)")
# plt.ylabel("Magnitude(dB)")
# plt.savefig(figurePath+"task5ExampleF.pdf")

#plt.show()