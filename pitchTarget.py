'''
音高人工标记
'''
from __future__ import print_function
import os;
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import librosa
import librosa.display
import scipy
from scipy.fftpack import fft,ifft
import queue
import math
from scipy.optimize import leastsq
from scipy import signal
from findPeaks import findpeaks
from scipy.interpolate import interp1d
import binascii

class1_path="guqin8/"
class1_list=os.listdir(class1_path)
class1_listLen=len(class1_list)
class_db="./class1.txt"
Fs=44100
nfft=int(4410)#窗口尺寸
hopLength=int(nfft)#步长
rmseS=1#是否显示瞬时能量
rmseDS=0#瞬时能量diff 
EES=1#是否显示谱熵
EEDS=1
showTestView=0#是否逐帧显示fft过程,需要把所有弹出窗口均关闭，然后关闭一个fft窗口 就会弹出下一个（只会这么整了）

#用于线性拟合的函数
def func(p,x):
    return p*x
def error(p,x,y):
    return (func(p,x)-y)*(func(p,x)-y) 

#归一化函数
def MaxMinNormalization(x,minv,maxv):
    Min=np.min(x)
    Max=np.max(x)
    y = (x - Min) / (Max - Min+0.0000000001)*(maxv-minv)+minv;
    return y

#目标标记程序流程
for index in range(0,class1_listLen):
    print(class1_list[index])
    stream=librosa.load(class1_path+class1_list[index],mono=False,sr=Fs)#以Fs重新采样
    x=stream[0]
    print('sampling rate:',stream[1])#采样率
    plt.plot(x[0]);plt.xlabel('sample'); plt.ylabel('amp');
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=hopLength, window=scipy.signal.hamming))
    plt.figure(figsize=(12, 4))
    fftForPitch=np.copy(speech_stft[0:np.int(nfft/Fs*4000)])#4000hz以下信号用于音高检测
    #plt.imshow(fftForPitch)
    librosa.display.specshow(fftForPitch,sr=Fs,hop_length=nfft)
    rmse = librosa.feature.rmse(y=x[0],S=None,frame_length=nfft, hop_length=hopLength, center=True, pad_mode='reflect')[0]
    times = librosa.frames_to_time(np.arange(len(rmse)),sr=Fs,hop_length=hopLength,n_fft=nfft)

    plt.figure(figsize=(12, 4))
    if rmseS==1:
        plt.plot(times, MaxMinNormalization((rmse),0,1),label='rmse_hop')
    #求功率熵
    speech_stft_Enp=speech_stft[1:-1]
    speech_stft_prob=speech_stft_Enp/np.sum(speech_stft_Enp,axis=0)
    EE=np.sum(-np.log(speech_stft_prob)*speech_stft_prob,axis=0)
    if EES==1:
        plt.plot(times, MaxMinNormalization(EE,0,1),label='EE')
    EEdiff=np.diff(MaxMinNormalization(EE,0,1))
    EEdiff=np.insert(EEdiff,0,0,None)
    if EEDS==1:
        #plt.plot(times, EEdiff)
        #plt.plot(times, EEdiffacc)
        test=EEdiff*MaxMinNormalization((rmse),0,1)
        plt.plot(times, test)
        EEdiff2=np.diff(MaxMinNormalization(EEdiff,0,1))
        EEdiff2=np.insert(EEdiff2,0,0,None)*MaxMinNormalization((rmse),0,1)
        #plt.plot(times, EEdiff2)
        plt.axhline(0, color='r', alpha=0.5)
    RMSEdiff =np.diff(MaxMinNormalization(rmse,0,1))
    RMSEdiff=np.insert(RMSEdiff,0,0,None)
    if rmseDS==1:
        plt.plot(times, RMSEdiff)
    plt.legend()
    speech_stft=np.transpose(speech_stft)
    plt.show()
    pre=0
    pitchs=[]
    for frame in np.arange(len(speech_stft)):
        if frame>-1:
            print([frame*nfft/Fs,"%.2f"%(frame/len(speech_stft)*100.0)]) #当前时刻
            dataClip=np.copy(speech_stft[frame])
            dataClip[0:int(30*nfft/Fs)]=0#清零30hz以下信号
            #线性内插重新采样
            processingX=np.arange(0,int(nfft/Fs*4000))#最大采集到4000Hz,不包括最大值，此处为尚未重采样的原始频谱
            processingY=dataClip[processingX]#重采样的fft
            lenProcessingX=len(processingX)#待处理频谱长度
            finterp=interp1d(processingX,processingY,kind='linear')#线性内插配置
            x_pred=np.linspace(0,processingX[lenProcessingX-1]*1.0,int(processingX[lenProcessingX-1]*441000/nfft)+1)
            maxProcessingX=x_pred[len(x_pred)-1]
            resampY=finterp(x_pred)
            lenResampY=len(resampY)
            #显示局部输入数据，便于人工标记
            
            
   
