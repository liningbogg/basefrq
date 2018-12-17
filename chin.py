import numpy as np
import librosa
from string import digits
import math
import re

class Chin:

    """
    Chin:
    用于处理古琴定音，音位分析，指法分析的类
    以十二平均律标记音高，hz为7条弦音高标识的主键，定调为do对应的note值
    音分散音 按音 泛音分别标记为 S A F
    相对徽位位置用来标记按音着弦点和有效弦长，另外有效弦长也可用相对弦长标识，最大相对弦长为1
    """
    noteslist = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    tonesList = [0,2,4,5,7,9,11]
    huiList=[0, 1.0/8, 1.0/6, 1.0/5, 1.0/4, 1.0/3, 2.0/5, 0.5, 3.0/5, 2.0/3, 3.0/4, 4.0/5, 5.0/5, 7.0/8, 1]
    fanyintimes = [8.0, 6.0, 5.0, 4.0, 3.0, 5.0, 2.0, 5.0, 3.0, 4.0, 5.0, 6.0, 8.0]


    @staticmethod
    def pos2hui(pos):
        count=0
        for hui in Chin.huiList:
            if pos>hui:
                count=count+1
                continue
            else:
                break
        start = Chin.huiList[count-1]
        end = Chin.huiList[count]
        return count-1+(pos-start)/(end-start)

    def note2tone(self, note):
        grade = int(re.findall("\d+",note)[0])-int(re.findall("\d+", self.do)[0])
        remove_digits = str.maketrans('', '', digits)
        reDo = note.translate(remove_digits)
        tone = 0
        try:
            tone=self.tones.index(reDo)+1
            print(tone)
        except Exception as e:
            print('err')
            return [0,0]
        return [tone,grade]

    def cal_tones(self):
        """
        确定唱名对应的音阶
        :return:
        """
        self.tones=[]
        remove_digits = str.maketrans('', '', digits)
        reDo = self.do.translate(remove_digits)
        initpos=Chin.noteslist.index(reDo)
        count=0
        for item in Chin.tonesList:
            item = item+initpos
            pos = item%12
            self.tones.append(Chin.noteslist[pos])
            count = count+1
        print(self.tones)

    @staticmethod
    def cal_position(basefrq, frq):
        """
        根据散音音高，待求位置的频率，求有效相对弦长，也就是着弦位点
        音高可以是相对音高，也可是绝对音高
        :param basefrq:散音基频
        :param frq:目标频率
        :return:着弦位点
        """
        return 1.0 * basefrq / frq

    @staticmethod
    def cal_notesposition(rightboundary):
        """
        计算12平均律相对音阶对应的位置，一般作为初始化的常量设置
        :param rightboundary: 琴弦右侧边界，对应最小有效相对弦长
        :return:12平均律音阶对应的位置
        """
        # pitches:计算各个音阶相对音高
        pitches = 2**np.arange(1 / 12, 4, 1 / 12)
        pos = []
        for pitch in pitches:
            poscan = 1.0/pitch
            if poscan>rightboundary:
                pos.append(poscan)

        return pos

    def cal_sanyinpred(self, pitch, thr):
        """
        匹配散音
        :param pitch: 待匹配音高
        :param thr: 匹配阈值
        :return:匹配结果
        """
        rs = []
        dist = abs(self.hzes - pitch)  # 音高距离
        candidate = np.argmin(dist)  # 候选散音
        relative = pitch / self.hzes[candidate]
        score = (relative - 1) / (2 ** (1.0 / 12) - 1)
        if abs(score) < thr:
            rs.append([candidate+1, score])
        return rs

    def cal_anyinstring(self, stringPitch, pitch, thr, spaceThr):
        """
        单一弦按音推测
        :param stringPitch 散音音高
        :param pitch:音高
        :param thr: 音准阈值
        :param spaceThr 绝对距离阈值
        :return:
        """

        positionR = self.cal_position(stringPitch, pitch) # 相对位置
        if positionR>0.96:
           return 0
        notesR=math.log(pitch/stringPitch, 2**(1/12)) # 相对散音音高
        candidateNoteR=np.round(notesR)   # 候选相对散音音高
        errR=notesR-candidateNoteR #相对散音音高误差
        if abs(errR) > thr:
            if Chin.pos2hui(positionR)>1.5:
                return -1.0*Chin.pos2hui(positionR)
            else:
                return 0
        else:
            candidate_positionR=1/((2**(1/12))**candidateNoteR)
            if abs(positionR-candidate_positionR)<spaceThr and Chin.pos2hui(positionR)>1.5:
                return Chin.pos2hui(positionR)
            else:
                return 0


    def cal_anyinpred(self, pitch, thr, spaceThr):
        """
        七弦按音推测
        :param pitch:音高
        :param thr:音位阈值
        :spaceThr：音位绝对位置阈值，反映手指精度
        :return:
        """
        rs = []
        for i in np.arange(7):
            anyin=self.cal_anyinstring(self.hzes[i], pitch, thr, spaceThr)
            if anyin != 0:
                rs.append([i+1 , anyin])
        return rs

    def cal_fanyinstring(self, stringpitch, pitch, thr):
        """
        获取指定弦泛音音位
        :param stringpitch:  散音频率
        :param pitch:  输入频率
        :param thr:  匹配误差阈值
        :param spacethr:  音位绝对距离误差阈值
        :return: 可能的音位
        """
        # 泛音由于5倍泛音 7倍泛音的关系，不采用2**1/12做误差分析

        times=pitch/stringpitch #频率比
        if times<1.5:
            return []
        candidate_time=np.round(times) #候选倍数
        err=times-np.round(times)
        rs=[]
        if abs(err)<thr:
            for i in np.arange(13):
                if Chin.fanyintimes[i]==candidate_time:
                    rs.append(i+1)
        return rs



    def cal_fanyinpred(self, pitch, thr):
        """
        预测可能的泛音音位
        :parweiam pitch:  音高
        :param thr:  note百分比误差阈值
        :param spacethr: 音位绝对距离阈值
        :return:返回可能的泛音音位
        """
        rs = []
        for i in np.arange(7):
            fanyin = self.cal_fanyinstring(self.hzes[i], pitch, thr)
            if fanyin !=  []:
                rs.append([i+1, fanyin])
        return rs


    def cal_possiblepos(self, pitches):
        """
        计算可能的音类及音位
        :param pitches:待解析的音高集合
        :return: 特定音高对应的可能的音位集合
        """
        number=len(pitches)
        possiblepos = []  # 结果
        formatStr = ""  # 格式化结果，用于打印
        for i in range(number):
            possiblepos.append([])
        thrsanyin=0.2
        thranyin=0.2
        thranyinspace=0.02/1.2
        thrfanyin=0.0875 #有待确定，确定这点比较难

        for i in np.arange(number):
            pitch=pitches[i]
            formatStr = formatStr+"%.2f\n" %pitch
            # 散音检测
            sanyinPred = self.cal_sanyinpred(pitch, thrsanyin)
            if sanyinPred != []:
                possiblepos[i].append(sanyinPred)
                formatStr = formatStr + "s:%d弦散音 e:%.2f\n" %(sanyinPred[0][0], sanyinPred[0][1])
            # 按音检测
            anyinPrep=self.cal_anyinpred(pitch, thranyin, thranyinspace)
            if anyinPrep != []:
                possiblepos[i].append(anyinPrep)
                formatStr = formatStr + "a:"
                for anyin in anyinPrep:
                    formatStr = formatStr + "%d弦%.2f徽  " % (anyin[0], anyin[1])
                formatStr = formatStr + "\n"
            # 泛音音位
            fanyinpred = self.cal_fanyinpred(pitch, thrfanyin)
            if fanyinpred != []:
                possiblepos[i].append(fanyinpred)
                formatStr = formatStr + "f:"
                for fanyin in fanyinpred:
                    for huiwei in fanyin[1]:
                        formatStr = formatStr + "%d弦%d徽  " % (fanyin[0], huiwei)
                formatStr = formatStr + "\n"
        return [possiblepos, formatStr]


    def __init__(self, **kw):
        """
        :param notes:
        七条弦依次对应的note（十二平均律标识）
        :param a4_hz:
        a4对应的频率
        :param do:
        唱名为do的note标识
        """
        self.notes = None
        self.do = None
        self.a4_hz = None
        self.hzes = None
        self.tones = None
        self.scaling = 1
        self.pos = self.cal_notesposition(0.125)
        for key in kw:
            try:
                if key == "notes":
                    setattr(self, key, kw[key])
                    string_num = len(self.notes)
                    if string_num != 7:
                        raise Exception("string number err:notes number err %d!" % string_num)
                    self.hzes=np.zeros(7)
                    for i in np.arange(string_num):
                        self.hzes[i] = librosa.note_to_hz(self.notes[i])*self.scaling
                if key == "hzes":
                    setattr(self, key, kw[key])
                    string_num = len(self.hzes)
                    if string_num != 7:
                        raise Exception("string number err:hzes number err %d!" % string_num)
                    self.notes=[None]*7
                    for i in np.arange(string_num):
                        self.notes[i] = librosa.hz_to_note(self.hzes[i], cents=True)
                if key == "a4_hz":
                    setattr(self, key, kw[key])
                    self.scaling = self.a4_hz / 440.0
            except Exception as e:
                print(e)

    def get_notes(self):
        """
        notes bean get
        :return: 返回notes
        """
        return np.copy(self.notes)

    def set_notes(self,notes):
        """
        notes bean set
        :param notes: 七弦音高的十二平均律标识设置，H代表不改变此前音高
        :return: None
        """
        self.notes = np.where(notes == "H", self.notes, notes)
        self.hzes = np.zeros(7)
        for i in np.arange(7):
            self.hzes[i] = librosa.note_to_hz(self.notes[i])*self.scaling

    def get_hzes(self):
        """
        :return:
        bean 返回hzes
        """
        return np.copy(self.hzes)

    def set_hzes(self, hzes):
        """
        设置七条弦音高，hz标识
        :param hzes: hz为-1表示不改变之前设置
        :return: None
        """
        self.hzes = np.where(hzes == -1, self.hzes, hzes)

    def get_ahz(self):
        return self.a_hz

    def set_ahz(self, a_hz):
        self.a4_hz=a_hz

    def get_do(self):
        return self.do

    def set_do(self, do):
        self.do = do
        self.cal_tones()


