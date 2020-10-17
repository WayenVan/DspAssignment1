#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:57:41 2020

@author: wayenvan
"""


"""import essential modules"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import wavfile
from enum import Enum

"""change font to fit Latex"""
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

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
def readWavefile(address):
    """read wave file and devide into two channel"""
    # check if the file is a wave
    assert os.path.splitext(address)[-1]==".wav"
    sampleRate, wav = wavfile.read(address)
    # make sure the wave is 2 channel
    assert len(wav.shape) == 2
    
    
    lchannel = wav[:,0]
    rchannel = wav[:,1]
    
    return (sampleRate, lchannel, rchannel)

def writeWavefile(address, sampleRate, lchannel, rchannel):
    """read wave file from folder"""
    #merge left and right channel
    data = np.hstack((lchannel[...,np.newaxis], rchannel[...,np.newaxis]))
    wavfile.write(address, sampleRate, data)


def subPlot(x, y, xlabel, ylabel, legend, title, xscale="linear", yscale="linear"):
    """plot each subplot in time domain """   
    plt.title(title)
    plt.plot(x, y, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.legend()
    plt.grid()

def wavePlotPSD(figure, x, lchannel, rchannel, legend="waveform"):
    """plot all channels of wave in time domain once"""
    xlabel="Frequency(Hz)"
    ylabel="Magnitude"
    
    plt.figure(figure,figsize=(20,10))
    plt.subplot(2,1,1)
    subPlot(x, lchannel, xlabel, ylabel, legend, title="Left channel PSD", xscale="log")
    plt.subplot(2,1,2)
    subPlot(x, rchannel, xlabel, ylabel, legend, title="Right channel PSD", xscale="log")
    plt.show()
    plt.savefig("./Output/"+figure+".pdf")
    
def wavePlotT(figure, x, lchannel, rchannel, legend="waveform"):
    """plot all channels of wave in time domain once"""
    xlabel="Time(s)"
    ylabel="Amplitude"
    
    plt.figure(figure,figsize=(20,10))
    plt.subplot(2,1,1)
    subPlot(x, lchannel, xlabel, ylabel, legend, title="Left channel time")
    plt.subplot(2,1,2)
    subPlot(x, rchannel, xlabel, ylabel, legend, title="Right channel time")
    plt.show()
    plt.savefig("./Output/"+figure+".pdf")

    
def wavePlotF(figure, xf, lchannelf, rchannelf, legend="waveform"):
    """plot all channels of signal in frequency dowmain once"""
    xlabel="Freqency(Hz)"
    ylabel="Amplitude(dB)"
    
    plt.figure(figure, figsize=(20,10))
    plt.subplot(2,1,1)
    subPlot(xf, lchannelf, xlabel, ylabel, legend, title="Left channel frequency ", xscale = "log")
    plt.subplot(2,1,2)
    subPlot(xf, rchannelf, xlabel, ylabel, legend,  title="Right channel frequency", xscale = "log")
    plt.show()
    plt.savefig("./Output/"+figure+".pdf")

def generateXf(sampleRate, N):
    """generateXf for frequeny domain"""
    return np.linspace(0.0, (N-1)*sampleRate/N, N)

def generateXt(sampleRate, N):
    """generateXt for time domain"""
    return np.linspace(0.0, (N-1)*1/sampleRate, N)
    
def mag2dB(yf):
    """ change magnitude into dB form """
    return 20*np.log10(yf)
    
def modifyWindow(w, startFreqency, endFreqency, sampleRate, value):
    """modify the window function into rectangular form"""
    beginPoint = int(startFreqency//(sampleRate/N))
    endPoint = int(endFreqency//(sampleRate/N))
    
    w[beginPoint:endPoint] = value
    w[-endPoint:-beginPoint] = value

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
    
    dtmfMin = 12
    dtmfMax = 12
    
    #calculate the peak point
    ind = np.argpartition(abs(rdataf), -3)[-3:]
    
    if((2/N*(abs(rdataf)[ind[0]])<minMagnitude) | (2/N*(abs(rdataf)[ind[1]])<minMagnitude)):
        return 'N'
    
    f1 = ind[0]*(sampleRate/N)
    f2 = ind[1]*(sampleRate/N)
    
    
    
    #start the for loop to check if the frequency meet the demand
    (flag1, index1) = findFrequencyBelong(f1, dtmfMin, dtmfMax, sampleRate)
    (flag2, index2) = findFrequencyBelong(f2, dtmfMin, dtmfMax, sampleRate)
    
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
    
    K = 0
    N = len(data)
    gap = 200
    T = 1/sampleRate
    
    preResult = 'N'
    seriesNumber = ''
    
    while gap-1+K*gap < N:
        result = detectOneDigitFromChunk(data[K*gap: gap-1+K*gap], sampleRate)
        if((preResult=='N') & (result != 'N')):
            print(K*gap*T, "-", (gap-1+K*gap)*T)
            seriesNumber = seriesNumber + result
        preResult = result
        K = K + 1
        
    return seriesNumber

"""main function """

inputWaveAddress = "./resources/example2.wav"
outputWaveAddress = "./Output/refinedVoice.wav"

(rate, lchannel, rchannel) = readWavefile(inputWaveAddress)

N = np.size(lchannel)
T = 1.0/rate
xt = generateXt(rate, N)
xf = generateXf(rate, N)

#plot time domain wave
#wavePlotT(xt, lchannel, rchannel)

"""start fft"""
#caculate fft
lchannelf = np.fft.fft(lchannel)
rchannelf = np.fft.fft(rchannel)

#calculate PSD
PSDlchannelf = np.abs(lchannelf)**2 / N
PSDrchannelf = np.abs(rchannelf)**2 / N

"""task3 refine the record"""
#generate window
w = np.ones(N)
modifyWindow(w, 200, 900, rate, 5)
modifyWindow(w, 6000, 10000, rate, 5)

lchannelfRefine = lchannelf*w
rchannelfRefine = rchannelf*w

lchannelRefine = np.fft.ifft(lchannelfRefine)
rchannelRefine = np.fft.ifft(rchannelfRefine)

"""task5"""
#load .dat file
dataI = np.loadtxt('./Resources/msc_matric_9.dat', usecols=(1), dtype=np.int16)

data = dataI
Fs2 = 1000 
N2 = len(data)
x2 = range(N2)
xt2 = generateXt(Fs2, N2)
xf2 = generateXf(Fs2, N2)
dataf = np.fft.fft(data)

series = autoDetectNumbers(data, Fs2)
print(series)

"""plot all figures"""
#plot frequency
#wavePlotF("frequencydomain", xf[0:N//2], mag2dB(2/N*np.abs(lchannelfRefine[0:N//2])), mag2dB(2/N*np.abs(rchannelfRefine[0:N//2])), legend="refined")
#wavePlotF("frequencydomain", xf[0:N//2], mag2dB(2/N*np.abs(lchannelf[0:N//2])), mag2dB(2/N*np.abs(rchannelf[0:N//2])), legend="unrefined")

#plot time domain wave form
#wavePlotT("timedomain", xt, lchannelRefine.astype(np.int16), rchannelRefine.astype(np.int16), legend="refined")
#wavePlotT("timedomain", xt, lchannel, rchannel, legend="unrefined")

#plot task5 wave
plt.figure(figsize=(20,10))
plt.plot(xt2, data)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(20,10))
plt.plot(xf2, mag2dB(abs(2/N2*dataf)))
plt.title("Frequency domain")
plt.xlabel("Freqency(Hz)")
plt.ylabel("Magnitude(dB)")
plt.show()

"""export the .wav file"""
writeWavefile(outputWaveAddress, rate, lchannelRefine.astype(np.int16), rchannelRefine.astype(np.int16))





