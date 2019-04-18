import matplotlib.pyplot as plt
import pandas
import numpy as np
import math
from keras.models import Sequential , load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,mean_squared_error

def create_datasets(dates,prices):
    train_size=int(0.80*len(dates))
    TrainX,TrainY=[],[]
    TestX,TestY=[],[]
    TrainX,TestX = dates[0:train_size] , dates[train_size:]
    TrainY,TestY = prices[0:train_size] , prices[train_size:]
            
    return TrainX,TrainY,TestX,TestY

dataset = pandas.read_csv('carriage_services.csv', usecols=[0,2], engine='python', skipfooter=3)
#print(dataset.values)
dates,prices = [],[]
for data in dataset.values:
    dates.append(int(data[0].split('-')[2]))
    prices.append(float(data[1]))
    #prices.append([float(data[1]),float(data[2]),float(data[3])])

#print( prices )

TrainX,TrainY,TestX,TestY=create_datasets(dates,prices)
#Multi-Layer-Perceptron
model = Sequential()
model.add(Dense(8,input_dim=1,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(1))
adam = Adam(lr=0.01)
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

model.fit(TrainX,TrainY,epochs=150,batch_size=2,verbose=2)

#Evaluate 
PredY = model.predict(TestX)

print('MSE_ => ',mean_squared_error(TestY,PredY))


#plt.plot(dataset['High'])
#plt.show()

