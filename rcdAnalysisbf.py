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
root_data_path = "/home/liningbo/文档/pyAudioAnalysis-master/tests/"
class1_path="guqin7/"
class2_path="guqin2/"
class1_list=os.listdir(root_data_path+class1_path)
class1_listLen=len(class1_list)
class_db="./class1.txt"
Fs=44100
nfft=int(4410)
rmseS=1
rmseDS=0
EES=0
EEDS=0
showTestView=1
def func(p,x):
    k,b=p
    return k*x+b
 
def error(p,x,y,s):
    return func(p,x)-y #x、y都是列表，故返回值也是个列表

def MaxMinNormalization(x,minv,maxv):
    Min=np.min(x)
    Max=np.max(x)
    y = (x - Min) / (Max - Min+0.0000000001)*(maxv-minv)+minv;
    return y
#递归算法求右侧波峰
def getNextPeaks(peaks,cepstrum,biasThr,distThr):
    #最小二乘法预判右侧紧邻波峰
    lenPeaks=len(peaks)#波峰数量
    if lenPeaks>1:
        harmonicIDs=np.arange(lenPeaks+1)+1#设置波峰ID
        p0=[peaks[lenPeaks-1]/lenPeaks,0]#初始化参数
        Xi=np.array(harmonicIDs[0:lenPeaks])
        Yi=np.array(peaks)
        s="test"
        para=leastsq(error,p0,args=(Xi,Yi,s)) #把error函数中除了p以外的参数打包到args中
        nextPeakPos=int(para[0][0]*harmonicIDs[lenPeaks]+para[0][1])#下一个峰位置预测
    else:
        nextPeakPos=2*peaks[0]        
    rad=int((nextPeakPos-peaks[lenPeaks-1])/2)#搜索波峰半径
    if nextPeakPos>(len(cepstrum)-rad):
        return 0;
    #print(nextPeakPos-rad,nextPeakPos+rad)
    nextMaxpos=np.argmax(cepstrum[nextPeakPos-rad:nextPeakPos+rad])+nextPeakPos-rad#下一个波峰检测到的位置
    bias=abs(nextMaxpos-nextPeakPos)/rad#计算波峰偏离
    #判断检测到打波峰位置跟预测的是否一致
    if bias<0.1:
        trough=np.min(cepstrum[peaks[lenPeaks-1]:nextMaxpos])#波谷检测
        dist=(cepstrum[nextMaxpos]-trough)/(cepstrum[peaks[lenPeaks-1]]-trough)#海拔高度检测
        if dist>0.20:#宽松判别条件
            peaks.append(nextMaxpos)
            getNextPeaks(peaks,cepstrum,biasThr,distThr)
    return 0
#第一个返回值是进行拟合并过滤的返回值 第二个返回值是原始返回值（用于调试）
def getPitch(dataClip,Fs,nfft):
    pitch=0
    dataClip[0:int(30*nfft/Fs)]=0
    print(frame*nfft/Fs) #当前时刻
    fftData=fft(dataClip)[0:int(len(dataClip)/2)]
    if showTestView: 
        plt.subplot(121)
        plt.plot(np.arange(len(dataClip)), dataClip)
    cepstrum=np.abs(fftData)*2/nfft
    if showTestView:
        plt.subplot(122)
        plt.plot(np.arange(len(cepstrum)), cepstrum)
    
    length=len(cepstrum)
    cutoff=int(Fs/2/800)
    peaks=[]
    pos=cutoff
    maxpos=np.argmax(cepstrum[pos:-1])+pos#最高峰位置
    #检测半峰
    simiPos=int(maxpos/2)
    rad=int(simiPos/2)
    simiMax=np.argmax(cepstrum[simiPos-rad:simiPos+rad])+simiPos-rad#半峰位置
    bias=abs(simiPos-simiMax)/rad#偏离度10%
    if bias<0.2:
        #半峰与最高峰之间的最小值
        simiMin=np.min(cepstrum[simiMax:maxpos])
        #与半峰最小值差值
        dist=cepstrum[simiMax]-simiMin
        #差值比例
        dist=dist/(cepstrum[maxpos]-simiMin)
        if(dist>0.66):
            #1 2
            #return Fs/2/simiMax//
            #四分检测
            
            fourPos1=int(simiMax/2)#四分之一峰
            p0=[fourPos1,0]
            Xi=np.array([2,4])
            Yi=np.array([simiMax,maxpos])
            s="test"
            para=leastsq(error,p0,args=(Xi,Yi,s)) #把error函数中除了p以外的参数打包到args中
            fourPos1=int(para[0][0]+para[0][1])
            fourPos2=int(para[0][0]*3+para[0][1])
            rad=int(fourPos1/2)
            
            fourMax1=np.argmax(cepstrum[fourPos1-rad:fourPos1+rad])+fourPos1-rad#四分峰位置1
            bias=abs(fourPos1-fourMax1)/rad#偏离度10%
            if bias<0.2:
                #半峰与四分峰之间的最小值
                fouMin1=np.min(cepstrum[fourMax1:simiMax])
                #与半峰最小值差值
                dist=cepstrum[fourMax1]-fouMin1
                #差值比例
                dist=dist/(cepstrum[simiMax]-fouMin1)
                if(dist>0.66):
                    #寻找四分3峰
                    fourMax2=np.argmax(cepstrum[fourPos2-rad:fourPos2+rad])+fourPos2-rad#四分峰位置1
                    bias=abs(fourPos2-fourMax2)/rad#偏离度10%
                    if bias<0.2:
                        #添加四分峰
                        peaks.append(fourMax1)
                        peaks.append(simiMax)
                        peaks.append(fourMax2)
            if len(peaks)<1:
                peaks.append(simiMax)
            
    #不存在半峰才检测三分峰         
    if len(peaks)<1:
        thd1=0#第一个三分峰的位置
        thd2=0#第二个三分峰的位置
        #三分峰值检测
        thrPos=int(maxpos*2/3)
        rad=int((maxpos-simiPos)/2)
        thrMax=np.argmax(cepstrum[thrPos-rad:thrPos+rad])+thrPos-rad#三分峰位置
        bias=abs(thrPos-thrMax)/rad#偏离度10%
        if bias<0.2:
            #2/3峰与最大峰之间的最小数
            thrMin=np.min(cepstrum[thrMax:maxpos])
            #与最小值差值
            dist=cepstrum[thrMax]-thrMin
            #差值比例
            dist=dist/(cepstrum[maxpos]-thrMin)
            if(dist>0.66):
                #return Fs/2/simiMax
                #寻找1 如果寻找不到1 则取消2
                thd1=thrMax
                #三分峰2检测 （做回归）
                p0=[maxpos/3,0]
                Xi=np.array([2,3])
                Yi=np.array([thd1,maxpos])
                s="test"
                para=leastsq(error,p0,args=(Xi,Yi,s)) #把error函数中除了p以外的参数打包到args中
                thrPos2=int(para[0][0]*1+para[0][1])
                rad=int((thd1-thrPos2)/2)
                #print(rad)
                #print(thrPos2)
                thrMax2=np.argmax(cepstrum[thrPos2-rad:thrPos2+rad])+thrPos2-rad#三分峰位置
                bias=abs(thrPos2-thrMax2)/rad#偏离度10%
                #print(bias)
                if bias<0.2:
                    #与2/3峰之间的最小值
                    thrMin2=np.min(cepstrum[thrMax2:thrMax])
                    #与最小值差值
                    dist=cepstrum[thrMax]-thrMin2
                    #差值比例
                    dist=dist/(cepstrum[thrMax2]-thrMin2)
                    if(dist>0.66):
                        thd2=thrMax2
                    else:
                        thd1=0
                        thd2=0 
                else:
                    #1:
                    #return Fs/2/maxpos
                    thd1=0
                    thd2=0
        if thd1>0:
             peaks.append(thd2)
             peaks.append(thd1)
             #print(thd1)
             #print(thd2)
    peaks.append(maxpos)

    #每做一次线性回归 找下一个峰
    getNextPeaks(peaks,cepstrum,0.1,0.66)#递归求右侧波峰
    lenPeaks=len(peaks)
    if showTestView:
        if lenPeaks>0:
            print(peaks)
        plt.show()
    if lenPeaks>1:
        #线性拟合求基频
        harmonicIDs=np.arange(lenPeaks)+1#设置波峰ID
        p0=[peaks[lenPeaks-1]/lenPeaks,0]#初始化参数
        Xi=np.array(harmonicIDs)
        Yi=np.array(peaks)
        s="test"
        para=leastsq(error,p0,args=(Xi,Yi,s)) #把error函数中除了p以外的参数打包到args中
        return [Fs/2/para[0][0],Fs/2/peaks[0]]
    else:
        return [0,0]


for index in range(0,class1_listLen):
    print(class1_list[index])
    stream=librosa.load(root_data_path + class1_path+class1_list[index],mono=False,sr=None)
    x=stream[0]    
    plt.plot(x[0]);plt.xlabel('sample'); plt.ylabel('amp');
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=nfft, window=scipy.signal.hamming))
    plt.figure(figsize=(12, 4))
    plt.imshow(speech_stft[0:np.int(nfft/Fs*4000)])
    rmse = librosa.feature.rmse(y=x[0],S=None,frame_length=nfft, hop_length=nfft, center=True, pad_mode='reflect')[0]
    times = librosa.frames_to_time(np.arange(len(rmse)),sr=Fs,hop_length=nfft,n_fft=nfft)
    plt.figure(figsize=(12, 4))
    if rmseS==1:
        plt.plot(times, MaxMinNormalization(rmse,0,1)) 
        #plt.axhline(0.001, color='r', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('RMSE')
        plt.axis('tight')
        plt.tight_layout()
    speech_stft_Enp=speech_stft
    speech_stft_prob=speech_stft_Enp/np.sum(speech_stft_Enp,axis=0)
    EE=np.sum(-np.log(speech_stft_prob)*speech_stft_prob,axis=0)
    if EES==1:
        plt.plot(times, MaxMinNormalization(EE,0,1))
        plt.xlabel('Time')
        plt.ylabel('EE')
        plt.axis('tight')
        plt.tight_layout()
    EEdiff=np.diff(MaxMinNormalization(EE,0,1))
    EEdiff=np.insert(EEdiff,0,0,None)
    if EEDS==1:
        plt.plot(times, EEdiff)
        plt.xlabel('Time')
        plt.ylabel('EEDiff')
        plt.axis('tight')
        plt.tight_layout()
    RMSEdiff =np.diff(MaxMinNormalization(rmse,0,1))
    RMSEdiff=np.insert(RMSEdiff,0,0,None)
    if rmseDS==1:
        plt.plot(times, RMSEdiff)
        plt.xlabel('Time')
        plt.ylabel('EEDiff')
        plt.axis('tight')
        plt.tight_layout()
   
    speech_stft=np.transpose(speech_stft)
    
    pl=plt.figure(figsize=(12, 8))
    pre=0
    pitchs=[]
    rawPitchs=[]
    for frame in np.arange(len(speech_stft)):
        dataClip=np.copy(speech_stft[frame])
        pitch=getPitch(dataClip,Fs,nfft)
        pitchs.append(pitch[0])
        rawPitchs.append(pitch[1])
    plt.plot(times,pitchs,label='pitch')
    plt.plot(times,rawPitchs,label='rawPitch')
    plt.plot(times, MaxMinNormalization(rmse,0,np.max(pitchs)),label='rmse')
    plt.plot(times, -MaxMinNormalization(EE,0,np.max(pitchs))/2,label='- EE')
    plt.axhline(0, color='r', alpha=0.5)
    plt.legend()
    plt.show()
