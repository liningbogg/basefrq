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
from time import sleep
from threading import Thread
import time
import baseFrqCombScan
import baseFrqComb

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
    rmse=MaxMinNormalization((rmse),0,1)
    plt.figure(figsize=(12, 4))
    if rmseS==1:
        plt.plot(times, rmse,label='rmse_hop')
    #求功率熵
    speech_stft_Enp=speech_stft[1:-1]
    speech_stft_prob=speech_stft_Enp/np.sum(speech_stft_Enp,axis=0)
    EE=np.sum(-np.log(speech_stft_prob)*speech_stft_prob,axis=0)
    EE=MaxMinNormalization(EE,0,1)
    if EES==1:
        plt.plot(times, EE,label='EE')
    EEdiff=np.diff(EE)
    EEdiff=np.insert(EEdiff,0,0,None)
    if EEDS==1:
        #plt.plot(times, EEdiff)
        #plt.plot(times, EEdiffacc)
        test=EEdiff*rmse
        plt.plot(times, test)
        EEdiff2=np.diff(EEdiff)
        EEdiff2=np.insert(EEdiff2,0,0,None)*rmse
        #plt.plot(times, EEdiff2)
        plt.axhline(0, color='r', alpha=0.5)
    RMSEdiff =np.diff(rmse)
    RMSEdiff=np.insert(RMSEdiff,0,0,None)
    if rmseDS==1:
        plt.plot(times, RMSEdiff)
    '''
    mergeEED=merge(EEdiff,rmse)[0]#融合区域后的EED
    mergeInfo=np.array(merge(EEdiff,rmse)[1])
    mergeamp=mergeInfo[:,2]
    mergeInit=mergeInfo[:,0]
    mergeInit=mergeInit.astype(np.int32)
    '''
    [mergeEED,mergeEEDINFO]=np.array(merge(EEdiff,rmse))
    
    #print(mergeEEDINFO)
    '''clipStop=mergeInfo[np.where(mergeamp>0.1)][:,0]
    clipStart=mergeInfo[np.where(mergeamp<-0.2 )][:,0]
    clipStart=clipStart.astype(np.int32)
    '''
    
    clipStop=[i for i in mergeEEDINFO  if (i[2]>0.15 ) ]
    clipStart=[i for i in mergeEEDINFO  if (i[2]<-0.2 and i[3]>0.2) ]
   
    if MergeEEDS==1:
        plt.plot(times, mergeEED,label='MEEDS')
    
    plt.legend()
    speech_stft=np.transpose(speech_stft)
    plt.show()
    pre=0
    pitchs=[]
    extendFrames=int(pitchExtend*Fs/nfft)#向前扩展的帧数
    plt.figure()
    speech_stft_pitch=np.copy(speech_stft)#求音高用短时傅里叶频谱
    #speech_stft_pitch=np.transpose(speech_stft_pitch)
    #print(len(speech_stft_pitch))
    #预先求一部分音高参考标签
    #用于标记的不去扫描线的音高
    referencePitch=[]
    referencePitchDeScanOri=[]
    #用于标记的去扫描线的音高
    '''for frm in np.arange(extendFrames+1):
        clipForPitch=np.copy(speech_stft_pitch[frm])
        referencePitchDeScanOri.append(baseFrqCombScan.getPitchDeScan(clipForPitch,Fs,nfft,0))
        referencePitch.append(baseFrqComb.getPitch(clipForPitch,Fs,nfft,0))'''

    
    #print(referencePitchDeScan)
    for frame in np.arange(len(speech_stft)):
        if frame>55:
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
            #初步设置显示2s以内的数据
            extendClips=np.arange(max(0,frame-extendFrames),frame+extendFrames)#延长的帧ID
            for extendFrm in np.arange(len(referencePitchDeScanOri),frame+extendFrames):
                clipForPitch=np.copy(speech_stft_pitch[extendFrm])
                referencePitchDeScanOri.append(baseFrqCombScan.getPitchDeScan(clipForPitch,Fs,nfft,0))
                referencePitch.append(baseFrqComb.getPitch(clipForPitch,Fs,nfft,0))
            referencePitchDeScan=[x[0] for x in referencePitchDeScanOri]
            referencePitchDeScanInput=[x[1] for x in referencePitchDeScanOri]
            referencePitchDeScanMedium=[x[2] for x in referencePitchDeScanOri]
            referenceRmse=np.copy(rmse[extendClips])
            referenceEE=np.copy(EE[extendClips])
            referenceMEED=np.copy(mergeEED[extendClips])
            referenceMEED=np.insert(referenceMEED,0,0,None)
            referenceTimes=librosa.frames_to_time(extendClips,sr=Fs,hop_length=hopLength,n_fft=nfft)-0.05
            plt.close()
            plt.subplot(221)
            plt.plot(referenceTimes, referenceRmse,label='rmse')
            plt.plot(referenceTimes, referenceEE,label='EE')
            plt.plot(referenceTimes, referenceMEED[0:len(referenceTimes)],label='MEED')
            plt.axvline((frame)*hopLength/Fs,color='r')
            plt.axhline(0,color='r')

            plt.annotate('%.2f'%  EE[frame], xy = (frame, EE[frame]), xytext = ((frame-0.5)*hopLength/Fs,EE[frame]))
            plt.annotate('%.2f'%  rmse[frame], xy = (frame, rmse[frame]), xytext = ((frame-0.5)*hopLength/Fs,rmse[frame]))
            plt.legend()
            plt.subplot(223)
            plt.axvline((frame)*hopLength/Fs,color='r')
            plt.axhline(0,color='r')
            currentClipStart=np.array([i[4] for i in clipStart  if (i[0]<frame+extendFrames and i[0]>frame-extendFrames) ])
            currentClipStart=currentClipStart.astype(np.int32)
            for startPos in currentClipStart:
                plt.axvline((startPos)*hopLength/Fs,color='b')
                plt.annotate('(%.2f,%.2f)'  %(startPos ,referencePitchDeScan[startPos]), xy = ((startPos-0.5)*hopLength/Fs, referencePitchDeScan[startPos]), xytext = ((startPos-0.5)*hopLength/Fs,referencePitchDeScan[startPos]))
            currentClipStop=np.array([i[4] for i in clipStop  if (i[0]<frame+extendFrames and i[0]>frame-extendFrames) ])
            currentClipStop=currentClipStop.astype(np.int32)
            for stopPos in currentClipStop:
                plt.axvline((stopPos)*hopLength/Fs,color='r')
            plt.plot(referenceTimes, referencePitchDeScan[max(0,frame-extendFrames):frame+extendFrames],label='pitchDeScan')
            plt.plot(referenceTimes, referencePitch[max(0,frame-extendFrames):frame+extendFrames],label='pitch')
            plt.legend()
            plt.subplot(222)
            plt.plot(np.arange(len(referencePitchDeScanInput[frame])),referencePitchDeScanInput[frame])
            plt.subplot(224)
            plt.plot(np.arange(len(referencePitchDeScanMedium[frame])),referencePitchDeScanMedium[frame])
            plt.show()
            pitchinfo=input("pitch:")
            
            
            
            
   
