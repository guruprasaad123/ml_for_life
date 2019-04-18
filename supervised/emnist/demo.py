# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv # opencv for image Manipulation
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
from keras import backend as K
import matplotlib.pyplot as plt

#K.set_image_dim_ordering('th')

def to_image_arr(path=""):
    img_arr = cv.imread('./hashy/'+path,cv.IMREAD_COLOR)
    img_arr_resized = cv.resize(img_arr,(32,32),interpolation=cv.INTER_LINEAR)
    return img_arr_resized

def create_label(data=None):
    list_ = [81,82,87]
    X = [to_image_arr(i) for i in data['path']]
    Y =[list_.index(i) for i in data['symbol_id'] if i is not None]
    #print("len Y : {}".format(len(Y)))
    #print("CLASSES {}".format(len(np.unique(Y))))

    return (np.array(X),Y)

data = pd.read_csv('./hashy/hasy-data-labels.csv',usecols=[0,1,2])

new_data = data[data.symbol_id.isin([81,82,87])]

print('data => ', new_data.shape ) 
(X,Y)=create_label(new_data)

Y=to_categorical(Y)
print('Y',Y.shape)
X_train , X_test , Y_train , Y_test  = train_test_split(X,Y,test_size=0.20,train_size=0.80)

ntrain=len(X_train)
ntest=len(X_test)
batch_size=4

model = models.Sequential()
model.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(32,32,3),padding="same"))
model.add(layers.MaxPooling2D((2,2) ,dim_ordering="th",padding="same"))
model.add(layers.Conv2D(32,(3,3),activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2,2),dim_ordering="th",padding="same"))
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(output_dim=3,activation='softmax'))

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
                              epochs=10,
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
