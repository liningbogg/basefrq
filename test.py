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
import pickle

np.set_printoptions(threshold=np.nan,linewidth=np.nan) 
class1_path="guqin11/"
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
fileDescan=open('test.txt','rb')
while True:
    try:
        lista=pickle.load(fileDescan)
        
        plt.plot(lista)
        plt.show()
        
        #print('hello')
    except EOFError:
        print('finished')
        break
   
