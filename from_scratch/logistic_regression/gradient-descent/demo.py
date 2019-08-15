import numpy as np
import pandas as pd
from collections import OrderedDict

def add_intercept(X):
 
    intercept = np.ones((X.shape[0],1))
    X= np.reshape(X,(-1,1))
    #print('intercept',intercept,X)
    return np.concatenate((intercept, X), axis=1)

def sigmoid(x):
    return 1/(1+np.exp(x))

def get_data():
    data = OrderedDict(
        amount_spent =  [50,  10, 20, 5,  95,  70,  100,  200, 0],
        send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1]
        )
    df = pd.DataFrame.from_dict(data) # creating a dataframe 
    X = df['amount_spent'].astype('float').values # converting the type to 'float'
    y = df['send_discount'].astype('float').values # converting the type to 'float'
    return (X,y) # returning the X , y

def gradient_descent_runner(X,y,learning_rate=0.01,epochs=10000):
    X = add_intercept(X)
    W = np.zeros(X.shape[1])
    print('m =>' ,X)

    for i in range(epochs):
        theta = np.dot(X,W)
        h = sigmoid(theta)
        gradient =  np.dot( X.T , h-y) / y.size 
        W = W- ( learning_rate * gradient ) 
        if i % 1000 == 0:
            print('Running : ',i)
    
    print('updated W : ',W)
        

def run():
    X , y= get_data()
    gradient_descent_runner(X,y)



if __name__ == "__main__":
    run()
