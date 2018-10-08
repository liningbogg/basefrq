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
#root_data_path = "/home/liningbo/文档/pyAudioAnalysis-master/tests/"
class1_path="guqin6/"
class2_path="guqin2/"#暂时没用到
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
#递归算法求二次频谱的右侧波峰
#offset：如果值为0 则说明是正常寻找波峰 如果offset>1 那么说明之前连续offset次没有找到相应位置的波峰 做多允许连续2个波峰位置不存在
def getNextPeaks(peaks,cepstrum,biasThr,distThr,offset):
    if offset==3:
        return 0
    #最小二乘法预判右侧紧邻波峰
    lenPeaks=len(peaks)#波峰数量
    if lenPeaks>1:
        #harmonicIDs=np.arange(lenPeaks+1)+1#设置波峰ID
        p0=peaks[lenPeaks-1][1]/lenPeaks#初始化参数
        Xi=np.array(peaks)[:,0]
        Yi=np.array(peaks)[:,1]
        para=leastsq(error,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
        nextPeakPos=int(para[0]*(offset+peaks[lenPeaks-1][0]+1))#下一个峰位置预测
    else:
        nextPeakPos=(2+offset)*peaks[0][1]
    rad=int((nextPeakPos-peaks[lenPeaks-1][1])/(offset+1)/2)#搜索波峰半径
    biasThrABS=max(7,rad*biasThr)
    #超出边界则完成搜索
    if nextPeakPos>(len(cepstrum)-rad):
        return 0;
    nextMaxpos=np.argmax(cepstrum[nextPeakPos-rad:nextPeakPos+rad])+nextPeakPos-rad#下一个波峰检测到的位置
    #print([peaks,offset,nextPeakPos,rad,abs(nextMaxpos-nextPeakPos),biasThrABS])
    bias=abs(nextMaxpos-nextPeakPos)#计算波峰偏离
    #判断检测到打波峰位置跟预测的是否一致
    if bias<biasThrABS:
        trough=np.min(cepstrum[int(peaks[lenPeaks-1][1]+offset*(nextMaxpos-peaks[lenPeaks-1][1])/(offset+1)):nextMaxpos])#波谷检测
        dist=(cepstrum[nextMaxpos]-trough)/(cepstrum[peaks[lenPeaks-1][1]]-trough)#海拔高度检测
        distThrABS=distThr*(0.7**offset)
        minValTest=np.min(cepstrum[int(nextMaxpos-rad):int(nextMaxpos+rad)])
        minPosTest=np.argmin(cepstrum[int(nextMaxpos-rad):int(nextMaxpos+rad)])+int(nextMaxpos-rad)
        if minPosTest>nextMaxpos:
            prot=(cepstrum[nextMaxpos]-minValTest)/(np.sum(cepstrum[nextMaxpos:minPosTest+1]-minValTest))*(minPosTest-nextMaxpos+1)
        else:
            prot=(cepstrum[nextMaxpos]-minValTest)/(np.sum(cepstrum[minPosTest:nextMaxpos+1]-minValTest))*(-minPosTest+nextMaxpos+1)
        #判断是否符合作为波峰的条件
        #宽松判别条件 此后要改为依据波峰检测结果确定是否为新的峰值（波峰检测算法已经列出，根据距离高度寻峰）
        if dist>distThrABS and prot>1.667:
            peaks.append([offset+peaks[len(peaks)-1][0]+1,nextMaxpos])
            if len(peaks)<10:
                getNextPeaks(peaks,cepstrum,biasThr,distThr,0)#找到10个波峰即可退出程序
        else:
            getNextPeaks(peaks,cepstrum,biasThr,distThr,offset+1)
    else:
        getNextPeaks(peaks,cepstrum,biasThr,distThr,offset+1)
    return 0
#第一个是根据fft波峰拟合的返回值 第二个返回值是进行拟合并过滤的返回值 第三个返回值是原始返回值（用于调试）
def getPitch(dataClip,Fs,nfft):
    pitch=0
    dataClip[0:int(30*nfft/Fs)]=0
    #print(frame*nfft/Fs) #当前时刻
    fftData=fft(dataClip)[0:int(len(dataClip)/2)]
    if showTestView: 
        plt.subplot(121)
        plt.plot(np.arange(len(dataClip)), dataClip)
    cepstrum=MaxMinNormalization(np.abs(fftData)*2/nfft,0,1)
    cutoff=int(Fs/2/1200)
    #cepstrum[0:cutoff]=0
    cepstrum=MaxMinNormalization(cepstrum,0,1)
    if showTestView:
        plt.subplot(122)
        plt.plot(np.arange(len(cepstrum)), cepstrum)
   
    length=len(cepstrum)
    peakind = signal.find_peaks_cwt(cepstrum, np.arange(1,20))
    peakind2=findpeaks(cepstrum, spacing=15, limit=0.05)
    if showTestView==1:
        print(peakind)
        print(peakind2)
    # 把 corlor 设置为空，通过edgecolors来控制颜色
    if showTestView==1:
        plt.scatter(peakind2, cepstrum[peakind2], color='', marker='o', edgecolors='r', s=100)
    
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
            p0=fourPos1
            Xi=np.array([2,4])
            Yi=np.array([simiMax,maxpos])
            para=leastsq(error,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
            fourPos1=int(para[0])
            fourPos2=int(para[0]*3)
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
                        peaks.append([len(peaks)+1,fourMax1])
                        peaks.append([len(peaks)+1,simiMax])
                        peaks.append([len(peaks)+1,fourMax2])
            if len(peaks)<1:
                peaks.append([len(peaks)+1,simiMax])
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
                p0=maxpos/3
                Xi=np.array([2,3])
                Yi=np.array([thd1,maxpos])
                para=leastsq(error,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
                thrPos2=int(para[0]*1)
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
             peaks.append([len(peaks)+1,thd2])
             peaks.append([len(peaks)+1,thd1])
             #print(thd1)
             #print(thd2)
    peaks.append([len(peaks)+1,maxpos])
    #每做一次线性回归 找下一个峰，直至找出右侧所有峰位置
    getNextPeaks(peaks,cepstrum,0.1,0.25,0)#递归求右侧波峰
    lenPeaks=len(peaks)
    if showTestView:
        if lenPeaks>0:
            print(peaks)
    rs=[]    
    if lenPeaks>1:
        #线性拟合求基频，根据找到的所有二次fft峰求二次fft的基频，此基频求倒数然后乘以采样率就是原始信号的基频，对原始信号的高频部分分辨率不足
        #，但原始信号高频部分在第一次fft中有好的频率分辨率，因此后期可以弥补
        #harmonicIDs=np.arange(lenPeaks)+1#设置波峰ID
        p0=peaks[lenPeaks-1][1]/lenPeaks#初始化参数
        Xi=np.array(peaks)[:,0]
        Yi=np.array(peaks)[:,1]
        para=leastsq(error,p0,args=(Xi,Yi)) #把error函数中除了p以外的参数打包到args中
        peaksfindWidth=int(nfft/2/para[0]*0.6)
        peakindofFFT=findpeaks(dataClip, spacing=peaksfindWidth, limit=max(dataClip[int(30*nfft/Fs):-1])/10)
        if showTestView==1:
            plt.subplot(121)
            plt.scatter(peakindofFFT, dataClip[peakindofFFT], color='', marker='o', edgecolors='r', s=100)
            plt.show()
        rs= [Fs/2/para[0],Fs/2/peaks[0][1]]
    else:
        if lenPeaks==1:
            if showTestView==1:
                plt.show()
            rs= [0,0]
        else:
            plt.show()
            rs= [0,0]
    return rs
#程序流程
for index in range(0,class1_listLen):
    print(class1_list[index])
    stream=librosa.load(class1_path+class1_list[index],mono=False,sr=None)
    x=stream[0]    
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
        
    speech_stft_Enp=speech_stft
    speech_stft_prob=speech_stft_Enp/np.sum(speech_stft_Enp,axis=0)
    EE=np.sum(-np.log(speech_stft_prob)*speech_stft_prob,axis=0)
    if EES==1:
        plt.plot(times, MaxMinNormalization(EE,0,1),label='EE')
    EEdiff=np.diff(MaxMinNormalization(EE,0,1))
    EEdiff=np.insert(EEdiff,0,0,None)
    EEdiffacc=acc(EEdiff)
    if EEDS==1:
        plt.plot(times, EEdiff)
        #plt.plot(times, EEdiffacc)
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
