import numpy as np
import pandas as pd
from collections import OrderedDict

def loss(h,y):
    return ( -y * np.log(h) - ( 1- y )*(np.log(1-y)) ).mean()

def add_intercept(X):
    intercept = np.ones((X.shape[0],1))
    X= np.reshape(X,(-1,1))
    #print('intercept',intercept,X)
    return np.concatenate((intercept, X), axis=1)

def predict(x,w):
    x = add_intercept(x)
    h = np.dot(x,w)
    return sigmoid(h).round()

def sigmoid(x):
    '''
    returns sigmoid h(x)= 1/(e^-x + 1) of the input x
    '''
    return 1/(1+np.exp(-x))


def check_for_convergence(beta_old,beta_new,tol=1e-3):
    '''
    Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    #calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)
    
    #if change hasn't reached the threshold and we have more iterations to go, keep training
    return not (np.any(coef_change>tol) )

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
    #print('m =>' ,X)

    for i in range(epochs):
        theta = np.dot(X,W)
        h = sigmoid(theta)
        gradient =  np.dot( X.T , h-y) / y.size 
        W_old = W
        W = W - ( learning_rate * gradient )
        if check_for_convergence(W_old,W):
            print('Converged @ ',i)
            break; 
        if i % 1000 == 0:
            print('Running : ',i,W,W_old)
    
    print('test : ',predict(np.array([[15],[155],[45]]),W))
        

def run():
    X , y= get_data()
    gradient_descent_runner(X,y)



if __name__ == "__main__":
    run()
