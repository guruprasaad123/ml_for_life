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
    X = df['amount_spent'].astype('float').values
    y = df['send_discount'].astype('float').values
    return (X,y) # returning the X , y

def gradient_descent_runner(X,y,learning_rate=0.001,epochs=10000):
    X = add_intercept(X)
    print('m =>' ,X)

def run():
    X , y= get_data()
    gradient_descent_runner(X,y)



if __name__ == "__main__":
    run()
