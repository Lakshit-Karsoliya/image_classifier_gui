from tkinter import *
from tkinter import ttk
import tkinter as tk
import cv2
import os
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Layer , Dense , Conv2D , Dropout , Flatten , MaxPooling2D
from keras.optimizers import Adam
import numpy as np
import shutil
import time
import threading
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)

def make_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return model
def prepare_data():
    x=[]
    y=[]
    f1 = e1.get()
    f2 = e2.get()
    c1 = os.listdir(f1)
    c2 = os.listdir(f2)
    for i in c1:
        im = cv2.imread(f'{f1}/{i}')
        x.append(im)
        y.append(0)
    for i in c2:
        im = cv2.imread(f'{f2}/{i}')
        x.append(im)
        y.append(1)
    x = np.array(x)
    y = np.array(y)
    shutil.rmtree(f1)
    shutil.rmtree(f2)
    return x,y
def predict_result(img,model):
    prd = model.predict(img)
    prob = np.amax(prd)
    
    result = np.argmax(prd)
    if prob < 0.75:
        result = 0
        prob = 0
    return result , prob
def gather_class_data_1():
    folder = e1.get()
    if folder == '':
        folder = 'class1'
    main_frame.pack_forget()
    loding_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
    pb.start()
    os.mkdir(folder)
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    count = 0
    while True:
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame_copy = frame.copy()
        cv2.imshow('image',frame_copy)
        frame_copy = cv2.resize(frame_copy,(100,100))
        cv2.imwrite(f'{folder}/img{count}.jpg',frame_copy)
        count+=1
        if count==100:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
        time.sleep(0.10)
    cv2.destroyAllWindows()
    pb.stop()
    loding_frame.pack_forget()
    main_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
def gather_class_data_2():
    folder = e2.get()
    if folder == '':
        folder = 'class2'
    main_frame.pack_forget()
    loding_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
    pb.start()
    os.mkdir(folder)
    cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    count = 0
    while True:
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame_copy = frame.copy()
        cv2.imshow('image',frame_copy)
        frame_copy = cv2.resize(frame_copy,(100,100))
        cv2.imwrite(f'{folder}/img{count}.jpg',frame_copy)
        count+=1
        if count == 100:
            break
        if cv2.waitKey(1) & 0xFF == 27:
            break
        time.sleep(0.1)
    cv2.destroyAllWindows()
    pb.stop()
    pb.stop()
    loding_frame.pack_forget()
    main_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
def lets_do_it():
    # f1 = 0
    # f2 = 1
    f1 = e1.get()
    f2 = e2.get()
    main_frame.pack_forget()
    loding_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
    pb.start()
    x,y = prepare_data()
    model = make_model()
    model.fit(x,y,batch_size=32,epochs=10,verbose=1,validation_data=(x,y))
    print('model_trainde_successfully')
    model.save('model.h5')
    print("model saved successfully")
    model = load_model('model.h5')
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame_copy = frame.copy()
        frame_copy = cv2.resize(frame_copy,(100,100))
        cv2.imwrite(f'img.jpg',frame_copy)
        img = cv2.imread('img.jpg')
        imp = np.array(img)
        x = []
        x.append(img)
        x = np.array(x)
        res , prob = predict_result(x,model)
        if res==0:
            res = f1
        else:
            res = f2
        cv2.putText(frame,f'press q to exit',(10,20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,f'{res} and {prob}',(10,50),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('image',frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.remove('model.h5')
            os.remove('img.jpg')
            break
        if cv2.waitKey(1) & 0xFF == 27:
            os.remove('model.h5')
            os.remove('img.jpg')
            break
        os.remove('img.jpg')
    cv2.destroyAllWindows()
    pb.stop()
    loding_frame.pack_forget()
    main_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)
    root.destroy()



root = Tk()
root.geometry('370x370')
root.resizable(0,0)
root.title('BinaryClassification')

main_frame = Frame(root)
loding_frame = Frame(root)
pb = ttk.Progressbar(loding_frame,orient='horizontal',mode='indeterminate',length=280)
pb.pack(pady=100)

title = Label(main_frame,text='Binary Image Classification')
title.grid(column=1,row=0, columnspan = 2,pady=30,padx=10)

lbc1 = Label(main_frame,text='Class 1 Name')
lbc1.grid(column=1,row=1,padx=10,pady=4)

e1 = Entry(main_frame,)
e1.grid(column=1,row=2,padx=10)
c1_btn = Button(main_frame,text='class_1',border=0,bg='gray',command=lambda:threading.Thread(target=gather_class_data_1).start())
c1_btn.grid(column=1,row=3,padx=10,pady=10)
lbc2 = Label(main_frame,text='Class 1 Name')
lbc2.grid(column=2,row=1,padx=10,pady=4)
e2 = Entry(main_frame)
e2.grid(column=2,row=2,padx=10)
c2_btn = Button(main_frame,text='class_2',border=0,bg='gray',command=lambda:threading.Thread(target=gather_class_data_2).start())
c2_btn.grid(column=2,row=3,padx=10,pady=10)
go = Button(main_frame,text = '<--START-->',border=0,bg='green',fg='white',command=lambda:threading.Thread(target=lets_do_it).start())
go.grid(column=1,row=4,padx=10,pady=70)


main_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

root.mainloop()