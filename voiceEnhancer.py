#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:20:00 2020

@author: wayenvan
"""

"""import essential modules"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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
    
def wavePlotT(figure, x, lchannel, rchannel, legend="waveform"):
    """plot all channels of wave in time domain once"""
    xlabel="Time(s)"
    ylabel="Amplitude"
    
    plt.figure(figure,figsize=(20,10))
    plt.subplot(2,1,1)
    subPlot(x, lchannel, xlabel, ylabel, legend, title="Left channel time")
    plt.subplot(2,1,2)
    subPlot(x, rchannel, xlabel, ylabel, legend, title="Right channel time")

    
def wavePlotF(figure, xf, lchannelf, rchannelf, legend="waveform"):
    """plot all channels of signal in frequency dowmain once"""
    xlabel="Freqency(Hz)"
    ylabel="Amplitude(dB)"
    
    plt.figure(figure, figsize=(20,10))
    plt.subplot(2,1,1)
    subPlot(xf, lchannelf, xlabel, ylabel, legend, title="Left channel frequency ", xscale = "log")
    plt.subplot(2,1,2)
    subPlot(xf, rchannelf, xlabel, ylabel, legend,  title="Right channel frequency", xscale = "log")
    
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
    N = len(w)
    
    beginPoint = int(startFreqency//(sampleRate/N))
    endPoint = int(endFreqency//(sampleRate/N))
    
    w[beginPoint:endPoint] = value
    w[-endPoint:-beginPoint] = value
    
"""main function """

inputWaveAddress = "./resources/original.wav"
outputWaveAddress = "./Output/improved.wav"
figurePath = "./Output/Figures/"

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

"""task4 refine the record"""
#generate window
w = np.ones(N)
modifyWindow(w, 120, 900, rate, 5)
modifyWindow(w, 6000, 10000, rate, 5)

lchannelfRefine = lchannelf*w
rchannelfRefine = rchannelf*w

lchannelRefine = np.fft.ifft(lchannelfRefine)
rchannelRefine = np.fft.ifft(rchannelfRefine)

"""plot and save all figures"""
wavePlotT("time domain Record", xt, lchannel, rchannel, legend="record")
#plt.savefig("./Output/Figures/recordT.pdf")

wavePlotF("frequency domain Record", xf[0:N//2], mag2dB(2/N*np.abs(lchannelf[0:N//2])), mag2dB(2/N*np.abs(rchannelf[0:N//2])), legend="unrefined")
#plt.savefig("./Output/Figures/recordF.pdf")

#plot time domain wave form
wavePlotT("timedomainReference", xt, lchannelRefine.astype(np.int16), rchannelRefine.astype(np.int16), legend="refined")
wavePlotT("timedomainReference", xt, lchannel, rchannel, legend="unrefined")
#plt.savefig(figurePath+"recordTR.pdf")

#plot frequency
wavePlotF("frequencydomainReference", xf[0:N//2], mag2dB(2/N*np.abs(lchannelfRefine[0:N//2])), mag2dB(2/N*np.abs(rchannelfRefine[0:N//2])), legend="refined")
wavePlotF("frequencydomainReference", xf[0:N//2], mag2dB(2/N*np.abs(lchannelf[0:N//2])), mag2dB(2/N*np.abs(rchannelf[0:N//2])), legend="unrefined")
#plt.savefig(figurePath+"recordFR.pdf")


#plot the window
#plt.figure(figsize=(20,10))
#plt.plot(xf[0:N//2], w[0:N//2])
#plt.xlabel("Frequency(Hz)")
#plt.xscale("log")
#plt.ylabel("Amplitude")
#plt.savefig(figurePath+"window.pdf")

plt.show()

"""export the .wav file"""
writeWavefile(outputWaveAddress, rate, lchannelRefine.astype(np.int16), rchannelRefine.astype(np.int16))