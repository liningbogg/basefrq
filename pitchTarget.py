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
import pickle
import pyaudio 
import wave

#播放波形
def playWave(data):
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32, channels=1, rate=Fs, output=True)
    stream.write(data,num_frames=len(data))
    stream.stop_stream()
    #暂停
    stream.close()
    #关闭
    p.terminate()

#取得指定格式的dir列表
def filterListDir(path,fmt):
    dirList=os.listdir(path)
    filteredList=[]
    for name in dirList:
        spilted=os.path.splitext(name)
        if(spilted[1]==fmt):
            filteredList.append(name)
    return filteredList
class1_path="guqin6/"
target_path="guqin10/"
class1_list=filterListDir(class1_path,'.flac')#暂时测试这一种格式
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
#依据文件添加缓存数据
def addChche(pitch,inputV,mediumV,initPos,length,file):
    for i in np.arange(initPos,initPos+length):
        try:
            listV=pickle.load(file)
            pitch[i]=listV[0]
            inputV[i]=listV[1]
            mediumV[i]=listV[2]
            
        except EOFError:
            print('文件结束！')
            break
   

#删去缓存
def deleteCache(pitch,inputV,mediumV,initPos,length):
    for i in np.arange(initPos,initPos+length):
        pitch[i]=[]
        inputV[i]=[]
        mediumV[i]=[]
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
            try:
                #当前区域结束,设置区域值
                maxRmse=max(rmse[currentInit:i])
                maxEEPos=np.argmax(abs(x[currentInit:i]))+currentInit
                x[currentInit:i]=currentSum        
                info.append([currentInit,i,currentSum,maxRmse,maxEEPos])
                currentSum=x[i]
                currentInit=i
            except Exception:
                maxRmse =0
                info.append([currentInit,len(src),currentSum,maxRmse,maxEEPos])
        else:
            currentSum=currentSum+x[i]     
    x[currentInit:-1]=currentSum#补充最后一组
    try:
        maxRmse=max(rmse[currentInit:i])
    except Exception:
        maxRmse =0
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
thrarta=0.15
thrartb=0.2
throp=0.15
for index in range(0,class1_listLen):
    print(class1_list[index])
    referencePitch=[]
    referencePitchInput=[]
    referencePitchMedium=[]
    referencePitchDeScan=[]
    referencePitchDeScanInput=[]
    referencePitchDeScanMedium=[]
    stream=librosa.load(class1_path+class1_list[index],mono=False,sr=None)#以Fs重新采样
    baseName=os.path.splitext(class1_list[index])[0]
    #标记记录文件名称
    logName=class1_path+baseName+'_%d'%Fs+'_%d'%nfft+'_log'+'.txt'
    #标记文件名称
    targetName=class1_path+baseName+'_%d'%Fs+'_%d'%nfft+'_target'+'.txt'
    #引入预处理文件前缀
    pitchPrepPathDeScan=class1_path+baseName+'_%d'%Fs+'_%d/'%nfft+'_%d'%Fs+'_%d'%nfft+'_descan_'
    pitchPrepPathComb=class1_path+baseName+'_%d'%Fs+'_%d/'%nfft+'_%d'%Fs+'_%d'%nfft+'_comb_'
    #读入标记记录文件
    logFile=open(logName,'w+')
    
    
    x=stream[0]
    print('sampling rate:',stream[1])#采样率
    plt.plot(x[0]);plt.xlabel('sample'); plt.ylabel('amp');
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=hopLength, window=scipy.signal.hamming))
    frameNum=len(speech_stft[0])
    for i in np.arange(frameNum):
        referencePitch.append([])
        referencePitchInput.append([])
        referencePitchMedium.append([])
        referencePitchDeScan.append([])
        referencePitchDeScanInput.append([])
        referencePitchDeScanMedium.append([])
    initFrame=logFile.readline()
    if len(initFrame)==0:
        initFrame=0
        logFile.write('0\n')
        logFile.flush()
        
    print(['初始位置:',initFrame])
    
    plt.figure(figsize=(12, 4))
    fftForPitch=np.copy(speech_stft[0:np.int(nfft/Fs*4000)])#4000hz以下信号用于音高检测
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
   
    [mergeEED,mergeEEDINFO]=np.array(merge(EEdiff,rmse))    
    
    clipStop=[i for i in mergeEEDINFO  if (i[2]>throp ) ]
    clipStart=[i for i in mergeEEDINFO  if (i[2]<(-1*thrarta) and i[3]>thrartb) ]
   
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
    framePerFile=int(60*Fs/nfft)#1分钟每个文件
    cacheFile=[]
    frame=0
    stopFile=math.ceil(len(speech_stft)*1.0/framePerFile)
    #target文件
    fileTarget=open(targetName,'ba')
    #pickle.dump(referencePitch, fileTarget)            
    #fileTarget.flush()
    
    while(frame<len(speech_stft)):
        print([frame*nfft/Fs,"%.2f"%(frame/len(speech_stft)*100.0)]) #当前时刻
        print(frame)  
        currentID=int(frame/framePerFile)
        #如果当前帧导致文件替换则更新先关音高缓存数据
        newSet=np.arange(currentID-1,currentID+2)#当前应该设置的缓存文件集合
        newSet=[i for i in newSet if (i>-1 and i<stopFile)]#应该添加的缓存文件
        addSet=[i for i in newSet if (i not in cacheFile and i>-1 and i<stopFile)]#应该添加的缓存文件
        
        if addSet!=[]:
            #添加缓存
            for i in addSet:
                initPos=i*framePerFile
                length=framePerFile
                file=open(pitchPrepPathComb+'%02d'%i+'.txt','rb')
                addChche(referencePitch,referencePitchInput,referencePitchMedium,initPos,length,file)#通过文件增加缓存并做校验
                file.close()
                file =open(pitchPrepPathDeScan+'%02d'%i+'.txt','rb')
                addChche(referencePitchDeScan,referencePitchDeScanInput,referencePitchDeScanMedium,initPos,length,file)#通过文件增加缓存并做校验
                file.close()
        deleteSet=[i for i in cacheFile if i not in newSet]#应该删去的缓存
        
        if deleteSet!=[]:
            #删除缓存
            for i in deleteSet:
                initPos=i*framePerFile
                length=framePerFile
                deleteCache(referencePitch,referencePitchInput,referencePitchMedium,initPos,length)#删除缓存
                deleteCache(referencePitchDeScan,referencePitchDeScanInput,referencePitchDeScanMedium,initPos,length)#删除缓存
        cacheFile= newSet
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
            #提前一格,防止起振阶段被忽略
            plt.axvline((startPos-1)*hopLength/Fs,color='b',ls="--")
            plt.annotate('(%.2f,%.2f)'%(startPos-1 ,referencePitchDeScan[startPos-1]),\
                         xy = ((startPos-0.5-1)*hopLength/Fs,\
                         referencePitchDeScan[startPos-1]),\
                         xytext = ((startPos-0.5-1)*hopLength/Fs,referencePitchDeScan[startPos-1]))
        currentClipStop=np.array([i[4] for i in clipStop  if (i[0]<frame+extendFrames and i[0]>frame-extendFrames) ])
        currentClipStop=currentClipStop.astype(np.int32)
        for stopPos in currentClipStop:
            plt.axvline((stopPos-1)*hopLength/Fs,color='r',ls="--")
        plt.plot(referenceTimes, referencePitchDeScan[max(0,frame-extendFrames):frame+extendFrames],label='pitchDeScan')
        plt.plot(referenceTimes, referencePitch[max(0,frame-extendFrames):frame+extendFrames],label='pitch')
        plt.legend()
        plt.subplot(222)
        plt.plot(np.arange(len(referencePitchDeScanInput[frame])),referencePitchDeScanInput[frame])
        plt.subplot(224)
        plt.plot(np.arange(len(referencePitchDeScanMedium[frame])),referencePitchDeScanMedium[frame])
        plt.show()
        

        pitchinfo=input("cmd:")
        '''switch={
            "dt":lambda frame:int(time*Fs/fft)
        }'''
        while True:
            try:
                cmds=pitchinfo.split(';')
                for cmd in cmds:
                    item =cmd.split()#命令行及其参数
                    cmdstr=item[0]#命令字
                    if cmdstr=="dt":
                        time=int(item[1])
                        if(time<=frame):
                            YESNO=input("设置时间小于等于当前时间，是否确认修改？是键入Y,否则N:")
                            if YESNO=="Y":
                                frame=time#调整当前帧位置
                            else:
                                print("放弃当前设置")
                        else:
                            frame=time#调整当前帧位置
                    if cmdstr=="thrart":
                        a=float(item[1])
                        b=float(item[2])
                        if a>0:
                            thrarta=a
                        if b>0:
                            thrartb=b
                        #重新计算起始位置
                        clipStart=[i for i in mergeEEDINFO  if (i[2]<(-1*thrarta) and i[3]>thrartb)]
                    if cmdstr=="throp":
                        thr=float(item[1])
                        if thr>0:
                            throp=thr
                        clipStop=[i for i in mergeEEDINFO  if (i[2]>throp ) ]
                    if cmdstr=="ef":
                        extendFrames=int(item[1])
                    if cmdstr=="pl":
                        start=int(item[1])
                        stop=int(item[2])
                        data=np.copy(x[0][start*nfft:stop*nfft])
                        print(len(data))
                        playWave(data)
                    if cmdstr=="pt":
                        time=int(item[1])
                        #如果time<0,认为自动移动一帧
                        if time<=frame:
                            time=frame+1
                            print("time小于当前帧，自动移动一帧。")
                        for i in np.arange(frame,time):
                            name=baseName
                            init=frame*nfft
                            lenClip=nfft
                            src=np.copy(speech_stft[i])
                            tar=item[2:]
                            tar=[float(p) for p in tar]
                            candidate=referencePitchDeScan[i]
                            tar=np.array(tar)
                            tar=np.where(tar<0.0,candidate,tar)#如果频率小于0则以候选频率替代
                            targetItem=[name,Fs,init,lenClip,src,tar]#待写入条目
                            print(targetItem)
                break
            except Exception as e:
                print(e)
                print('请重新输入:')
                pass
                break
            
            
            
            
   
