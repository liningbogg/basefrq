from tkinter import *   #引用Tk模块
root = Tk()             #初始化Tk()
root.title("音高标记程序")#标题
root.geometry('800x600')#分辨率
label = Label(root, text="音频参考信息图形", bg="red", font=("Arial", 12), width=0, height=0)#音频信息标签
label.pack()  #这里的side可以赋值为LEFT  RTGHT TOP  BOTTOM
frm = Frame(root)#总矩形区域
#上方布局
frm_Top = Frame(frm)

frm_Top.pack(side=TOP)
frm.pack()
root.mainloop()         #进入消息循环

