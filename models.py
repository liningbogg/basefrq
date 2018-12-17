from django.db import models


# Create your models here.
class Clip(models.Model):
    """
    title:曲名
    startingPos:数据起始位置
    length:数据长度
    timestamp:时间戳
    src:数据源（fft）
    tar:标签
    """
    title = models.CharField(max_length=255)
    startingPos = models.IntegerField()
    length = models.IntegerField()
    timestamp = models.DateTimeField()
    src = models.BinaryField()
    tar = models.BinaryField()
    anote = models.CharField(max_length=255)
    nfft = models.IntegerField()


class Tone(models.Model):
    """
    pos:起始帧
    lengh:长度
    pitch：音高
    note:十二平均律标注
    tone:唱名 5-3 6+2 7 1+1 1-1
    anote:指法注释 s2 f7h9 a6h7.94
    """

    title = models.CharField(max_length=255)
    pos = models.IntegerField()
    length = models.IntegerField()
    pitch = models.FloatField()
    note = models.CharField(max_length=16)
    tone = models.CharField(max_length=16)
    anote = models.CharField(max_length=255)


class Wave(models.Model):
    title = models.CharField(max_length=255)
    waveFile = models.CharField(max_length=255)
    frameNum = models.IntegerField()
    notes = models.CharField(max_length=255)
    do = models.CharField(max_length=16)
    fs = models.IntegerField()


class Log(models.Model):
    """
    target操作记录
    title:与操作相关的曲目名称
    content:记录内容
    timestamp：时间戳
    """
    title = models.CharField(max_length=255)
    content = models.CharField(max_length=255)
    timestamp = models.DateTimeField()


class MarkedPhrase(models.Model):
    """
    需要关注的音乐片段
    title:乐曲名称
    mark:标注
    """
    title = models.CharField(max_length=255)
    mark = models.CharField(max_length=255)
    start = models.FloatField()
    length = models.FloatField()
