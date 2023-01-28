# image_classifier_gui
# requirments
<code>pip install opencv-python</code><br>
<code>pip install keras</code><br>
<code>pip install numpy</code><br>

# instructions

1.first type class 1 name if you leave it blank default value is class_1<br>
2.click on class_1 button a window opens and take 100 images<br>
3.type class 2 name default value is class_2<br>
4.click on class_2 button again 100 images taken<br>
5.ckick on start button it will take some time to train model after traning is done a cv2 window open and classify data frames in real time<br>

# working
class 1 and class 2 contains 100 images taken from webcam <br>
***if clicking 100 pictures takes too long then remove <code>time.sleep(0.10)</code> or tweak its value*** <br>
these images scaled down to 100x100 pixel images have three channels <br>
after training the model cv2 window capture frames in real time and show output on same cv2 window<br>



# how model looks like
<code>
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
</code>
