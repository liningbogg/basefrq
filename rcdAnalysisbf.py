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


#root_data_path = "/home/liningbo/文档/pyAudioAnalysis-master/tests/"
class1_path="guqin8/"
class2_path="guqin2/"#暂时没用到
class1_list=os.listdir(class1_path)
class1_listLen=len(class1_list)
class_db="./class1.txt"
Fs=44100
nfft=int(4410)#窗口尺寸
hopLength=int(nfft)#步长
rmseS=1#是否显示瞬时能量
rmseDS=0#瞬时能量diff 
EES=0#是否显示谱熵
EEDS=1
STE=0
showTestView=0#是否逐帧显示fft过程,需要把所有弹出窗口均关闭，然后关闭一个fft窗口 就会弹出下一个（只会这么整了）
#短时瞬间能量（没有用上，用了RMSE）
def ShortTimeEnergy(seq,nfft,hopLen):   
    frameNum=math.ceil(len(seq)/hopLen)#计算帧数
    rs=np.zeros(frameNum)
    for i in range(0,frameNum):
        rs[i]=np.sum(seq[i*hopLen:(i*hopLen+nfft)]**2)
    rs=MaxMinNormalization(rs,0,1)
    return rs
#用于线性拟合的函数
def func(p,x):
    return p*x
def error(p,x,y):
    return (func(p,x)-y)*(func(p,x)-y) 
#累加函数
def acc(arr):
    rs=np.copy(arr)
    tmp=0
    for i in range(0,len(arr)):
       rs[i]=sum(arr[i-5:i])
    return rs
#归一化函数
def MaxMinNormalization(x,minv,maxv):
    Min=np.min(x)
    Max=np.max(x)
    y = (x - Min) / (Max - Min+0.0000000001)*(maxv-minv)+minv;
    return y
#次峰限幅
def subpeakAmpLimiting(dataClip,space,limit):
    peaks=findpeaks(dataClip, spacing=space, limit=max(dataClip)*limit)
    dots=np.copy(dataClip[peaks])
    dots[np.argmax(dots)]=0
    maxval=max(dots)
    print(maxval)
    #如果等于0 说明只有一个峰
    if maxval>0:
        dataClip=np.where(dataClip<maxval,dataClip,maxval)
    return dataClip   
#获得范围内最大峰
def getNearPeaks(trueTrans,combTransPeaks,peaks,n,combThr,preBaseFrq,decThr):
    if showTestView:
        print(['preBaseFrq',preBaseFrq])
        print(['n',n])
        print(['combThr',combThr])
        print(['combTransPeaks',combTransPeaks])
    
    a=np.where(combTransPeaks>(n-combThr))[0]
    b=np.where(combTransPeaks<(n+combThr))[0]
    selectPeaks=[v for v in a if v in b]
    if showTestView:
        print(['a',a])
        print(['b',b])
        print(['test',selectPeaks])
    
    if len(selectPeaks)>0 :
        candidacy =combTransPeaks[selectPeaks[np.argmax(peaks[selectPeaks])]]
        if showTestView:
            print(['selectPeaks[np.argmax(peaks[selectPeaks])]',selectPeaks[np.argmax(peaks[selectPeaks])]])
            print(['trueTrans[candidacy ]',trueTrans[candidacy ]])
            print(['trueTrans[preBaseFrq]',trueTrans[preBaseFrq]])
        if (trueTrans[candidacy ]/trueTrans[preBaseFrq])>decThr:
            return candidacy
        else:
            return -1
    else:
        return -1
def getScanDot(deSamp,height):
    biasShift=deSamp-height
    lenDesamp=len(deSamp)
    a=biasShift[0:lenDesamp-2]
    b=biasShift[1:lenDesamp-1]
    c=a*b
    jiaodian=np.where(c<=0)
    #print(height)
    #print(len(deSamp))
    x1=jiaodian[0][0]
    x2=x1+1
    y1=deSamp[x1]
    y2=deSamp[x2]
    x=((height-y1)/(y2-y1)+x1)
    return [x,height]
#横向扫描求轮廓
def getBaseLineFromScan(deSamp,num):
    minval=min(deSamp)#最小值
    maxval=np.mean(deSamp[16:26])#最大值
    intercept=(maxval-minval)/1000.0
    heights=np.arange(minval+intercept,maxval,intercept)
    x=[]
    y=[]
    pos=-1
    for i in heights:
        dot=getScanDot(deSamp[0:pos],i)
        x.append(dot[0]*10.0)
        y.append(dot[1])
        pos=int(dot[0])+20
    x.append(0)
    y.append(0)
    x.append(num)
    y.append(0)
    #print(type(dots))
    #print(dots)
    x=x
    finterp=interp1d(x,y,kind='linear')#线性内插配置
    x_pred=np.arange(0,num,1)
    resampY=finterp(x_pred)
    
    return resampY
def getPitch(dataClip,Fs,nfft):
    pitch=0
    dataClip[0:int(30*nfft/Fs)]=0
    dataClip=subpeakAmpLimiting(dataClip,int(30.0/Fs*nfft),0.1) #次峰限幅
    if showTestView: 
        plt.subplot(231)
        plt.plot(np.arange(len(dataClip)), dataClip,label='amp-frq')
    lowCutoff=int(40*441000.0/Fs)#最低截止频率对应的坐标
    highCutoff=int(1400*441000.0/Fs)#最高截止频率对应的坐标
    peakSearchPixes=int(3*441000/Fs)#寻峰间距
    peakSearchAmp=0.1#寻峰高度
    #线性内插重新采样
    processingX=np.arange(0,int(nfft/Fs*4000))#最大采集到4000Hz,不包括最大值，此处为尚未重采样的原始频谱
    processingY=dataClip[processingX]#重采样的fft
    lenProcessingX=len(processingX)#待处理频谱长度
    finterp=interp1d(processingX,processingY,kind='linear')#线性内插配置
    x_pred=np.linspace(0,processingX[lenProcessingX-1]*1.0,int(processingX[lenProcessingX-1]*441000/nfft)+1)
    maxProcessingX=x_pred[len(x_pred)-1]
    resampY=finterp(x_pred)
    lenResampY=len(resampY)
    #print(len(resampY))
    if showTestView==1:
        plt.subplot(232)
        plt.plot(np.arange(len(resampY)), resampY,label='rs_Amp-frq')
    maxResampY=max(resampY[lowCutoff:-1])/2#待测频率内的最大值
    #测试梳状变换
    #pixes=10
    num=highCutoff#栅栏变换后的长度
    combTrans=np.zeros(num)#存放栅栏变换的结果
    indexComb=np.arange(lowCutoff,highCutoff,0.1)#栅栏变换索引
    for k in np.arange(1,highCutoff,1):
        combTrans[k]=np.sum([resampY[i] for i in np.arange(0,lenResampY-1,k)])
    if showTestView==1:
        plt.subplot(233)
        plt.plot(np.arange(len(combTrans)), combTrans,label='combTrans')
    deSamp=[combTrans[m] for m in np.arange(0,len(combTrans),10)]#10倍数降采样
    deSamp[0]=1000
    if showTestView==1:
        plt.subplot(234)
        plt.plot(np.arange(len(deSamp)), deSamp,label='DsCombTrans')
    baseLine=getBaseLineFromScan(deSamp,num)#通过扫描线算法求基准曲线
    if showTestView==1:
        plt.subplot(235)
        plt.plot(np.arange(len(baseLine)), baseLine,label='baseline')
    trueTrans=combTrans-baseLine
    trueTrans[0:lowCutoff]=0
    if showTestView==1:
        plt.subplot(236)
        plt.plot(np.arange(len(trueTrans)), trueTrans,label='trueTrans')
    pitch=max(trueTrans)/sum(dataClip)
    combTransPeaks=findpeaks(trueTrans, spacing=peakSearchPixes, limit=max(trueTrans)*peakSearchAmp)
    peaks=trueTrans[combTransPeaks]
    maxindex=np.argmax(peaks)
    maxfrq=combTransPeaks[maxindex]
    #寻找1hz以内的最大的峰
    combThr=6*441000/Fs
    decThr=0.75#递减阈值
    preBaseFrq=maxfrq
    for n in range(2,10,1):
        newfrq=getNearPeaks(trueTrans,combTransPeaks,peaks,n*maxfrq,combThr,preBaseFrq,decThr)
        if(newfrq>0):
            preBaseFrq=newfrq
        else:
            continue
    pitch=preBaseFrq;
    if showTestView==1:
        plt.scatter(combTransPeaks, trueTrans[combTransPeaks], color='', marker='o', edgecolors='r', s=100)
        plt.show()
        
    return pitch/10.0
#程序流程
for index in range(0,class1_listLen):
    print(class1_list[index])
    stream=librosa.load(class1_path+class1_list[index],mono=False,sr=None)
    x=stream[0]
    print(stream[1])
    plt.plot(x[0]);plt.xlabel('sample'); plt.ylabel('amp');
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=hopLength, window=scipy.signal.hamming))
    plt.figure(figsize=(12, 4))
    plt.imshow(speech_stft[0:np.int(nfft/Fs*4000)])
    rmse = librosa.feature.rmse(y=x[0],S=None,frame_length=nfft, hop_length=hopLength, center=True, pad_mode='reflect')[0]
    times = librosa.frames_to_time(np.arange(len(rmse)),sr=Fs,hop_length=hopLength,n_fft=nfft)
    rmse2 = librosa.feature.rmse(y=x[0],S=None,frame_length=nfft, hop_length=nfft, center=True, pad_mode='reflect')[0]
    times2 = librosa.frames_to_time(np.arange(len(rmse2)),sr=Fs,hop_length=nfft,n_fft=nfft)
    plt.figure(figsize=(12, 4))
    if rmseS==1:
        plt.plot(times, MaxMinNormalization((rmse),0,1),label='rmse_hop')
        plt.plot(times2, MaxMinNormalization((rmse2),0,1),label='rmse')
        
    speech_stft_Enp=speech_stft[1:-1]
    speech_stft_prob=speech_stft_Enp/np.sum(speech_stft_Enp,axis=0)
    EE=np.sum(-np.log(speech_stft_prob)*speech_stft_prob,axis=0)
    if EES==1:
        plt.plot(times, MaxMinNormalization(EE,0,1),label='EE')
    EEdiff=np.diff(MaxMinNormalization(EE,0,1))
    EEdiff=np.insert(EEdiff,0,0,None)
    EEdiffacc=acc(EEdiff)
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
    ste=ShortTimeEnergy(x[0],nfft,hopLength)
    if STE==1:
        plt.plot(times, ste,label='STE')
    plt.legend()
    speech_stft=np.transpose(speech_stft)
    pl=plt.figure(figsize=(12, 8))
    pre=0
    pitchs=[]
    for frame in np.arange(len(speech_stft)):
        if frame>-1:
            print([frame*nfft/Fs,"%.2f"%(frame/len(speech_stft)*100.0)]) #当前时刻
            dataClip=np.copy(speech_stft[frame])
            pitch=getPitch(dataClip,Fs,nfft)
            pitchs.append(pitch)
    plt.plot(times,pitchs,label='pitch')
    plt.plot(times, MaxMinNormalization(rmse,0,np.max(pitchs)),label='rmse')
    plt.plot(times, -MaxMinNormalization(EE,0,np.max(pitchs))/2,label='- EE')
    plt.axhline(0, color='r', alpha=0.5)
    plt.legend()
    plt.show()
