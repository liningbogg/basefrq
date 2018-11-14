'''
音高人工标记
音高参考信息预处理
为了节约标记时间

'''
from __future__ import print_function
import os
import shutil
from datetime import datetime 
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
from time import sleep
from threading import Thread
import time
import baseFrqCombScan
import baseFrqComb

class1_path="guqin8/"
class1_list=os.listdir(class1_path)
class1_listLen=len(class1_list)
targetPath="guqin9/"
class_db="./class1.txt"
Fs=44100
nfft=int(4410)#窗口尺寸
hopLength=int(nfft)#步长

rmseS=1#是否显示瞬时能量
rmseDS=0#瞬时能量diff 
EES=1#是否显示谱熵
EEDS=1
MergeEEDS=1#融合区域后的EEDS
showTestView=0#是否逐帧显示fft过程,需要把所有弹出窗口均关闭，然后关闭一个fft窗口 就会弹出下一个（只会这么整了）
pitchExtend=4#为了标注音高延申的数据长度，单位秒


#用于积分的累积求和
def merge(src,rmse):
    x=np.copy(src)
    length=len(x)
    currentSum=0
    currentInit=0
    maxRmse=0
    maxEEPos=0
    info=[]
    for i in np.arange(length):
        if (currentSum*x[i])<0:
            #当前区域结束,设置区域值
            maxRmse=max(rmse[currentInit:i])
            maxEEPos=np.argmax(abs(x[currentInit:i]))+currentInit
            x[currentInit:i]=currentSum
            
            info.append([currentInit,i,currentSum,maxRmse,maxEEPos])
            currentSum=x[i]
            currentInit=i
            
        else:
            currentSum=currentSum+x[i]
            
    x[currentInit:-1]=currentSum#补充最后一组
    maxRmse=max(rmse[currentInit:i])
    info.append([currentInit,len(src),currentSum,maxRmse,maxEEPos])
    return [x,info]    

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
    startTime=datetime.now() 
    baseName=targetPath+os.path.splitext(class1_list[index])[0]
    stream=librosa.load(class1_path+class1_list[index],mono=False,sr=Fs)#以Fs重新采样
    
    x=stream[0]
    print('sampling rate:',stream[1])#采样率
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=hopLength, window=scipy.signal.hamming))
    speech_stft=np.transpose(speech_stft)
    referencePitch=[]
    referencePitchDeScan=[]
    #用于标记的去扫描线的音高
    fileDescanName=baseName+'_%d'%Fs+'_%d'%nfft+'_descan'+'.txt'
    fileCombName=baseName+'_%d'%Fs+'_%d'%nfft+'_comb'+'.txt'
    for frame in np.arange(len(speech_stft)):
        if frame>-1:
            print([frame*nfft/Fs,"%.2f"%(frame/len(speech_stft)*100.0)]) #当前时刻
            dataClip=np.copy(speech_stft[frame])
            dataClip[0:int(30*nfft/Fs)]=0#清零30hz以下信号
            referencePitchDeScan.append(baseFrqCombScan.getPitchDeScan \
                                        (dataClip,Fs,nfft,showTestView))
            referencePitch.append(baseFrqComb.getPitch(dataClip,Fs,nfft,showTestView))
    
    fileDescan=open(fileDescanName,'w')
    fileDescan.write(str(referencePitchDeScan))
    fileDescan.close()
    print(fileDescanName+' 音高参考信息写入完毕')
    fileComb=open(fileCombName,'w')
    fileComb.write(str(referencePitch))
    fileComb.close()
    print(fileCombName+' 音高参考信息写入完毕')
    endTime=datetime.now()
    print('用时%d;'%(endTime-startTime).seconds \
          +'帧数:%d;'%len(speech_stft) \
          +'spf:%f;'%((endTime-startTime).seconds*1.0/len(speech_stft)) \
          +'speed:%f'%((endTime-startTime).seconds*1.0/(len(speech_stft))*Fs/nfft))
    shutil.move(class1_path+class1_list[index],targetPath+class1_list[index])        
            
            
   
