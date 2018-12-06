import numpy as np
import librosa

class Chin:

    """
    Chin:
    用于处理古琴定音，音位分析，指法分析的类
    以十二平均律标记音高，hz为7条弦音高标识的主键，定调为do对应的note值
    音分散音 按音 泛音分别标记为 S A F
    相对徽位位置用来标记按音着弦点和有效弦长，另外有效弦长也可用相对弦长标识，最大相对弦长为1
    """
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
        self.scaling = 1
        self.cal_notesposition(0.125)
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
        self.do=do


