import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib as mlt
import matplotlib.pyplot as plt

def loss(h,y):
    return ( -y * np.log(h) - ( 1- y )*(np.log(1-y)) ).mean()

def add_intercept(X):
    intercept = np.ones((X.shape[0],1))
    #X= np.reshape(X,(-1,1))
    #print('intercept',intercept,X)
    return np.concatenate((intercept, X), axis=1)

def predict(x,w):
    x = add_intercept(x)
    h = np.dot(x,w)
    return sigmoid(h).round()

def sign(x,w):
    x = add_intercept(x)
    h = np.dot(x,w)
    return sigmoid(h)

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
        amount_spent =  [50,  10, 20, 5,  65,  70,  80,  81, 1],
        send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1]
        )
    df = pd.DataFrame.from_dict(data) # creating a dataframe 
    X = df['amount_spent'].astype('float').values # converting the type to 'float'
    y = df['send_discount'].astype('float').values # converting the type to 'float'
    return (X,y) # returning the X , y

def load_data(url):
    df=pd.read_csv(url,header=None);
    return ( df.iloc[:,:-1] , df.iloc[:,-1])

def gradient_descent_runner(X,y,learning_rate=0.01,epochs=10000):
    #X = add_intercept(X)
    W = np.zeros(X.shape[1])
    #print('m =>' ,X)

    for i in range(epochs):
        theta = np.dot(X,W)
        h = sigmoid(theta)
        gradient =  np.dot( X.T , h-y) / y.size 
        W_old = W
        W = W - ( learning_rate * gradient )
        if check_for_convergence(W_old,W):
            W = W_old
            print('Converged @ ',i)
            break; 
        if i % 1000 == 0:
            print('Running : ',i,W,W_old)

    return (W)
        

def run():
    X , y = load_data('./marks.txt')
    #X,y = get_data()
    W = gradient_descent_runner(X,y)
    t = np.arange(0,X.shape[0])

    plt.figure()
    X1 = X[y==0]
    X2 = X[y==1]
    #print( X1 )
    #print( X2 )
    print( W )

    plt.scatter( X1[0] , X1[1] , label="O's" )
    plt.scatter( X2[0] , X2[1] , label="1's" )

    #plt.plot(np.dot(add_intercept(X),W))
    # intercept = np.dot(add_intercept(X),W )
    # pre = predict(X,W)
    # sign_ = sign(X,W)
    # for i in range(pre.shape[0]):
    #     print("X = {} , X.W = {} , W = {} , X = {} , sign = {} , predict = {} ".format(
    #         X[i] ,
    #         intercept[i] , 
    #         sign_[i] / X[i] ,
    #         intercept[i] / X[i],
    #         sign_[i],
    #         pre[i] ) )
    print(  )

    W = np.array([-6.13154479 , 0.60748623 ,  0.15197772])

    X_extreme = np.array([np.min(X[0]), np.max(X[1])])
    
    #Y_extreme = - (W[1] * X_extreme)) / W[1] + 1/W[1]
    Y_extreme = -( (W[0] * X_extreme ) + 1) / W[1]
    print( X_extreme, Y_extreme )
    
    plt.plot( X_extreme , Y_extreme )
    plt.xlabel('Amount spent (X)',fontsize=14)
    plt.ylabel('( Y )',fontsize=14,rotation=0)
    plt.axis([0,100,0,100])

    #plt.axvline(.5, color='black')
    
    plt.legend(loc="upper left")
    plt.show()





if __name__ == "__main__":
    run()
