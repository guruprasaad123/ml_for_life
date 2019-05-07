# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv for image Manipulation
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
from array import array
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping
import matplotlib as plt

print(os.listdir("../input/data/natural_images/"))

classes = os.listdir("../input/data/natural_images/")

def get_class(url=None):
    _list = url.split('/')
    _class = _list[len(_list)-1].split('_')[0]
    return _class

def get_index(url=None,array=[]):
    _class=get_class(url)
    index = array.index(_class)
    return index

def humane_or_not(path="",model=None):
    img_arr = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(150,150),interpolation = cv2.INTER_LINEAR)
    index = model.predict_classes(img_arr)
    if classes[index] == 'person':
        return 'Humane'
    else:
        return 'Non-Humane'
    
    
    
def create_labels(classes=None,root="../input/data/natural_images/" ):
    result_image = []
    for i in range(len(classes)):
        images =[root+classes[i]+'/'+x for x in os.listdir(root+classes[i])]
        result_image=result_image + images
        #print("classes = {} , len = {} ,index[0] {}".format(classes[i],len(images),images[0]))
    
    random.shuffle(result_image)
    Y = np.array([get_index(i,classes) for i in result_image])
    X=[cv2.resize(cv2.imread(i,cv2.IMREAD_COLOR),(150,150),interpolation = cv2.INTER_LINEAR) for i in result_image]
    #print('img ',cv2.imread())
    print('Y ',type(Y))

    print('cat => ',Y[0])
    return (np.array(X),Y)


(X,Y)= create_labels(classes)
print('Y',Y.shape)
print('X',X.shape)
Y=to_categorical(Y)
X_train , X_test , Y_train , Y_test  = train_test_split(X,Y,test_size=0.20,train_size=0.80)

print(X_train.shape,Y_train.shape,Y_train[0])

print(X_test.shape,Y_test.shape,Y_test[0])


ntrain=len(X_train)
ntest=len(X_test)
batch_size=32
# Any results you write to the current directory are saved as output.
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(8,activation='softmax'))


model.compile(loss="categorical_crossentropy",optimizer=optimizers.Adam(),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True
                                )
                                
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train,Y_train,batch_size=batch_size)
test_generator = test_datagen.flow(X_test,Y_test,batch_size=batch_size)

callbacks = [EarlyStopping(monitor="acc",patience=5,mode='max')]
history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain,
                              epochs=128,
                              validation_data=test_generator,
                              validation_steps=ntest,
                              callbacks=callbacks
                                )
model.save_weights('model_weights.h5')
model.save('model.h5')

acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss=history.history['val_loss']

epochs= range(1,len(acc)+1)

plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()

'''
humane_or_not() method should be enough to classify whether the image is Humane or Not

'''