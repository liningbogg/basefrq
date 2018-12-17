'''
音高人工标记
'''

from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import scipy
import math
from scipy import signal
from findPeaks import findpeaks    
from scipy.interpolate import interp1d
import sys
import baseFrqCombScan
import pickle
import pyaudio
from datetime import datetime
import zlib
from chin import Chin
import django
from django.db.models import Max
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pitch.settings")  # project_name 项目名称django.setup()
django.setup()
from target.models import Clip
from target.models import Wave
from target.models import Tone
from target.models import Log
from target.models import MarkedPhrase


#播放波形
def playWave(data):
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32, channels=1, rate=Fs, output=True)
    stream.write(data,num_frames=len(data))
    stream.stop_stream()
    stream.close()
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
class1_path="../data/guqin6/"
target_path="../data/guqin10/"
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
toneShowWidth = 5  # 音高标记显示宽度
#依据文件添加缓存数据
def addChche(pitch,inputV,mediumV,initPos,length,file):
    for i in np.arange(initPos,initPos+length):
        try:
            listV=pickle.load(file)
            pitch[i]=listV[0]
            inputV[i]=listV[1]
            mediumV[i]=listV[2]
            
        except EOFError as e:
            print('文件结束！%d' %i)
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
#filter by basefrq
def filterByBasefrq(src, basefrq, width):
    peaksPos=findpeaks(src, spacing=50, limit=max(src)*0.05)
    peaks=src[peaksPos]  # 峰值大小
    tar=np.copy(src)
    num=min(int(len(src)/basefrq),30)
    for i in np.arange(num):
        frq=i*basefrq
        tar[frq-width:frq+width]=min(src[frq-width],src[frq+width])
    return tar
#目标标记程序流程
thrarta=0.15
thrartb=0.2
throp=0.15
# 默认基频过滤参数
defaultFltWidth = int(300)  # 默认频率过滤宽度 30hz
thrResPow = 0.2  # 过滤残余能量阈值 50%

lastestPos=[]
chin = Chin()  # notes=['c2', 'd2', 'f2', 'g2', 'a2', 'c3', 'd3']
index = 0
while True:
    # 选择曲目
    for i in range(0, class1_listLen):
        candidateName = os.path.splitext(class1_list[i])[0]
        candidateFrame = 0
        candidateCent = 0
        if Clip.objects.filter(title=os.path.splitext(class1_list[i])[0]).count() == 0:
            candidateFrame = 0
        else:
            candidateFrame = Clip.objects.all().aggregate(Max('startingPos'))['startingPos__max']
        # 计算进度
        setWave = Wave.objects.filter(title=candidateName)
        if setWave.count()>0:
            candidateFrameNum = setWave[0].frameNum  # 待测总帧数
            candidateCent = candidateFrame/candidateFrameNum*1.0
        print("[%02d]%s:%.2f" % (i, candidateName, candidateCent))
    songID = input("选择曲目:")
    index = int(songID)
    # 正式标记流程

    print(class1_list[index])
    referencePitch=[]
    referencePitchInput=[]
    referencePitchMedium=[]
    referencePitchDeScan=[]
    referencePitchDeScanInput=[]
    referencePitchDeScanMedium=[]
    songName = class1_path + class1_list[index]
    stream = librosa.load(songName, mono=False, sr=None)#以Fs重新采样
    baseName = os.path.splitext(class1_list[index])[0]

    #引入预处理文件前缀
    pitchPrepPathDeScan=class1_path+baseName+'_%d'%Fs+'_%d/'%nfft+'_%d'%Fs+'_%d'%nfft+'_descan_'
    pitchPrepPathComb=class1_path+baseName+'_%d'%Fs+'_%d/'%nfft+'_%d'%Fs+'_%d'%nfft+'_comb_'
    #读入标记记录文件
    x=stream[0]
    print('sampling rate:',stream[1])#采样率
    plt.plot(x[0])
    plt.xlabel('sample')
    plt.ylabel('amp')
    speech_stft,phase = librosa.magphase(librosa.stft(x[0], n_fft=nfft, hop_length=hopLength, window=scipy.signal.hamming))
    frameNum=len(speech_stft[0])

    if Clip.objects.filter(title=baseName).count() == 0:
        frame = 0
    else:
        print(Clip.objects.all().aggregate(Max('startingPos')))
        # 更新pos
        frame = Clip.objects.all().aggregate(Max('startingPos'))['startingPos__max'] + 1

    wavenum = Wave.objects.filter(title=baseName).count()
    print("test"+os.path.abspath(songName))
    if wavenum == 0:
        waveItem = Wave(title=baseName, waveFile=os.path.abspath(songName), frameNum=frameNum, fs=Fs)
        waveItem.save()
    wave = Wave.objects.get(title=baseName)
    if wave.notes != "":
        print(wave.notes)
        notesstr = input("是否保留notes，是则y，否则直接键入notes：")
        if notesstr!="y":
            wave.notes=notesstr
            wave.save()
    else:
        notesstr=input("键入notes：")
        wave.notes = notesstr
        wave.save(update_fields=["notes"])

    if wave.do != "":
        print(wave.do)
        dostr = input("是否保留do，是则y，否则直接键入do：")
        if dostr!="y":
            wave.do=dostr
            wave.save()
    else:
        dostr=input("键入do：")
        wave.do = dostr
        wave.save(update_fields=["do"])

    notesList = wave.notes. split(" ")
    chin.set_notes(notes=notesList)  # 设置notes
    chin.set_do(wave.do)  # 设置曲调

    for i in np.arange(frameNum):
        referencePitch.append([])
        referencePitchInput.append([])
        referencePitchMedium.append([])
        referencePitchDeScan.append([])
        referencePitchDeScanInput.append([])
        referencePitchDeScanMedium.append([])
    tarArray=np.ones((frameNum, 5))

    print(['初始位置:', frame])

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
    speech_stft = np.transpose(speech_stft)
    plt.show()
    pre=0
    pitchs=[]
    extendFrames=int(pitchExtend*Fs/nfft)#向前扩展的帧数
    plt.figure()
    speech_stft_pitch=np.copy(speech_stft)#求音高用短时傅里叶频谱
    framePerFile=int(60*Fs/nfft)#1分钟每个文件
    cacheFile=[]
    stopFile=math.ceil(len(speech_stft)*1.0/framePerFile)
    # target文件
    isfilterBybasefrq=False
    filterBasefrq=[]
    filterWidth=[]
    tmpShow=False
    while(frame<len(speech_stft)):
        tmpShow=False
        print("时刻:%.2f 进度:%.2f" % (frame*nfft/Fs, frame/len(speech_stft)*100.0))  # 当前时刻
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
        #从数据库获取标记的主音高
        candidate_clips = Clip.objects.filter(startingPos__range=(max(0,frame-extendFrames), frame+extendFrames)) #参考音高条目
        for candidate_clip in candidate_clips:
            candidate_tarstr = candidate_clip.tar #原始tar数据
            candidate_tar = pickle.loads(candidate_tarstr)
            candidate_pos = candidate_clip.startingPos
            tarArray[candidate_pos] = candidate_tar[0] # 更新要显示的标记主音

        plt.plot(referenceTimes, tarArray[max(0,frame-extendFrames):frame+extendFrames,0],label='tar0')
        plt.legend()
        plt.subplot(222)
        plt.plot(np.arange(len(referencePitchDeScanInput[frame])),referencePitchDeScanInput[frame])
        plt.subplot(224)
        plt.plot(np.arange(len(referencePitchDeScanMedium[frame])),referencePitchDeScanMedium[frame])
        resPitch = []  # 残余基频
        if isfilterBybasefrq==True:
            # 待过滤数据
            referencePitchDeScanInputFilter=np.copy(referencePitchDeScanInput[frame])
            for i in np.arange(len(filterBasefrq)):
                # 过滤基频
                referencePitchDeScanInputFilter=filterByBasefrq(referencePitchDeScanInputFilter,filterBasefrq[i],filterWidth[i])
                # 显示过滤后的残余
                plt.subplot(222)
                plt.plot(np.arange(len(referencePitchDeScanInputFilter)),referencePitchDeScanInputFilter,label='filter%d'%i)
                # 显示残余基频
                test=baseFrqCombScan.getPitchDeScan(referencePitchDeScanInputFilter,Fs,Fs*10,0)
                resPitch.append(test[0])
                print("残余基频%d：%.2f" % (i+1, test[0]))

        else:  # 默认过滤
            # 待过滤数据
            referencePitchDeScanInputFilter = np.copy(referencePitchDeScanInput[frame])
            resPow = sum(referencePitchDeScanInputFilter)  # 残余能量
            resCent = 1.0  # 残余比例
            fltPitch = referencePitchDeScan[frame]  # 默认音高
            index = 1  # 残余索引
            pre=1.0
            while fltPitch>40:
                # 过滤基频 (频率乘以10是因为分辨率为0.1hz)
                referencePitchDeScanInputFilter = \
                    filterByBasefrq(referencePitchDeScanInputFilter, int(fltPitch*10), defaultFltWidth)
                # 显示过滤后的残余
                plt.subplot(222)
                plt.plot(np.arange(len(referencePitchDeScanInputFilter)), referencePitchDeScanInputFilter,
                         label='filter%d' % index)
                resCent = sum(referencePitchDeScanInputFilter) / resPow
                # 显示残余基频
                test = baseFrqCombScan.getPitchDeScan(referencePitchDeScanInputFilter, Fs, Fs * 10, 0)
                if (pre-resCent)<thrResPow:
                    break
                print("残余基频%d：%.2f notes:%s  残余:%.2f"\
                      % (index, test[0], librosa.hz_to_note(test[0]*chin.scaling, cents=True), resCent))
                fltPitch = test[0]  # 更新过滤基音
                if resCent>thrResPow:
                    resPitch.append(fltPitch)
                index = index+1
                pre = resCent
        if isfilterBybasefrq==True:
            for pitchFlt in filterBasefrq:
                pitchFlt = pitchFlt/10.0
                print("默认基频 :%.2f notes:%s" % (pitchFlt, librosa.hz_to_note(pitchFlt * chin.scaling, cents=True)))
                print(chin.cal_possiblepos([pitchFlt])[1])
                isfilterBybasefrq = False
                filterBasefrq = []
                filterWidth = []
        else:
            if referencePitchDeScan[frame] > 40:
                print("默认基频 :%.2f notes:%s"\
                    % (referencePitchDeScan[frame],librosa.hz_to_note(referencePitchDeScan[frame]*chin.scaling, cents=True)))
                print(chin.cal_possiblepos([referencePitchDeScan[frame]])[1])

        if resPitch != []:
            str_yinwei = chin.cal_possiblepos(resPitch)[1]
            set_yinwei = chin.cal_possiblepos(resPitch)[0]
            print(str_yinwei)

        # 显示前后5s的tone标注
        toneShowFramesNum = int((Fs*1.0/nfft)*toneShowWidth)  # 前后扩展的帧数
        toneShow = Tone.objects.filter(title=baseName, pos__range=(frame-toneShowFramesNum, frame+toneShowFramesNum))
        showstr="历史音高标记:"
        for toneitem in toneShow:
            showstr = showstr+"%s,%s,%s |" % (toneitem.tone, toneitem.note, toneitem.pos)
        print(showstr)
        plt.legend()
        plt.show()

        pitchinfo = input("cmd:")
        Log(title=baseName, content=pitchinfo, timestamp=datetime.now()).save()
        while True:
            try:
                cmds=pitchinfo.split(';')

                for cmd in cmds:
                    item =cmd.split()#命令行及其参数
                    cmdstr=item[0]#命令字
                    if cmdstr=="mt":
                        time = int(item[1])
                        frame = time  # 调整当前帧位置
                        tmpShow=True
                    if cmdstr=="dt":
                        time=int(item[1])
                        if(time<=frame):
                            YESNO=input("设置时间小于等于当前时间，是否确认修改？是键入Y,否则N:")
                            if YESNO == "Y":
                                frame = time#调整当前帧位置
                            else:
                                print("放弃当前设置")
                        else:
                            frame=time#调整当前帧位置
                        Clip.objects.filter(startingPos__gte=frame).delete()
                        tmpShow = True
                    if cmdstr == "thrart":
                        a=float(item[1])
                        b=float(item[2])
                        if a > 0:
                            thrarta = a
                        if b > 0:
                            thrartb = b
                        #重新计算起始位置
                        clipStart = [i for i in mergeEEDINFO if (i[2]<(-1*thrarta) and i[3] > thrartb)]
                    if cmdstr=="throp":
                        thr=float(item[1])
                        if thr>0:
                            throp=thr
                        clipStop=[i for i in mergeEEDINFO  if (i[2]>throp ) ]
                    if cmdstr=="ef":
                        extendFrames=int(item[1])
                    if cmdstr=="tw":  # tone width
                        toneShowWidth = float(item[1])
                    if cmdstr=="pl":
                        start=int(item[1])
                        stop=int(item[2])
                        data=np.copy(x[0][start*nfft:stop*nfft])
                        playWave(data)
                    if cmdstr == "mark":
                        start = int(item[1])
                        stop = int(item[2])
                        length = (stop-start)*1.0*nfft/Fs  # 单位是时间
                        start = start*nfft*1.0/Fs
                        markstr = item[3]
                        MarkedPhrase(title=baseName, start=start, length=length, mark=markstr).save()

                    if cmdstr == "pt":
                        time = int(item[1])
                        #如果time<0,认为自动移动一帧
                        lastestPos = []
                        if time <= frame:
                            time = frame+1
                            print("time小于当前帧，自动移动一帧。")
                        for i in np.arange(frame, time):
                            name = baseName
                            init = frame
                            lenClip = 1
                            src = np.copy(speech_stft[i])
                            tar = item[2:]
                            tar = [float(p) for p in tar]
                            candidate = referencePitchDeScan[i]
                            tar = np.array(tar)
                            tar = np.where(tar < 0.0, candidate, tar)#如果频率小于0则以候选频率替代
                            # 写入数据库
                            srcstr = pickle.dumps(src)
                            tarstr = pickle.dumps(tar)
                            dbitem = Clip(title=name, startingPos=init, length=lenClip, src=srcstr, tar=tarstr,
                                        timestamp=datetime.now(), nfft=nfft)
                            lastestPos.append(init)
                            dbitem.save()
                            frame = frame+1
                    if cmdstr == "anote":
                        if lastestPos!=[]:
                            for pos in lastestPos:
                                lastItem = Clip.objects.get(startingPos=pos)
                                if len(item)>1:
                                    lastItem.anote = item[1]
                                else:
                                    lastItem.anote = ""
                                lastItem.save()
                        else:
                            print("没有最新条目")

                    if cmdstr == "flt":
                        filterBasefrq=[]
                        filterWidth=[]
                        for i in np.arange(1,len(item)):
                            if i%2 == 1:
                                filterBasefrq.append(int(float(item[i])*10))
                            else:
                                filterWidth.append(int(float(item[i])*10))
                        if len(filterBasefrq) != len(filterWidth) or len(filterBasefrq)<1:
                            break
                        isfilterBybasefrq = True

                    # 设置唱名条目
                    if cmdstr == "tone":
                        startTone = int(item[1])  # tone 起始位置
                        length = int(item[2])-startTone  # 不包括item[2]
                        pitchTone = float(item[3])  # 这里都不用-1
                        noteTone = librosa.hz_to_note(pitchTone/chin.scaling)  # 用于测量tone的note，不求百分数
                        tone = chin.note2tone(noteTone)  # 计算音高 ，因为程序编写费时间，不提供直接设置tone的方式
                        tonestr = '%d/%d' % (tone[0], tone[1])  # tone[0] 是音高， tone[1]是grade
                        if len(item) == 5:
                            anoteTone = item[4]
                        else:
                            anoteTone = ""
                        Tone(title=baseName, pos=startTone, length=length,\
                             pitch=pitchTone, note=librosa.hz_to_note(pitchTone/chin.scaling, cents=True),\
                             tone=tonestr, anote=anoteTone).save()

                    # 删除指定的唱名条目
                    if cmdstr == "dtone":
                        deleteTime = int(item[1])  # 删除时刻
                        Tone.objects.filter(title=baseName, pos__gte=deleteTime).delete()  # 删除time后的条目

                    # 描述可能的音位
                    if cmdstr == "desc":
                        pitchDesc = float(item[1])
                        print(chin.cal_possiblepos([pitchDesc])[1])

                    if cmdstr == "exit":

                        sys.exit(0)

                break
            except Exception as e:
                print(e)
                print('请重新输入:')
                pass
                break
        # 更新pos
        if tmpShow == False:
            lastest = Clip.objects.all()

            frame = lastest.aggregate(Max('startingPos'))['startingPos__max']
            if frame is not None:
                frame = frame+1
            else:
                frame = 0
            print(["test", frame])

            
            
            
   
