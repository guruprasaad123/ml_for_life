import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib as mlt
import matplotlib.pyplot as plt
from scipy import optimize

def get_data():
    data = OrderedDict(
        amount_spent =  [50,  10, 20, 5,  65,  70,  80,  81, 1],
        send_discount = [0,   1,  1,  1,  0,   0,   0,    0,   1]
        )
    df = pd.DataFrame.from_dict(data) # creating a dataframe 
    X = df['amount_spent'].astype('float').values # converting the type to 'float'
    y = df['send_discount'].astype('float').values # converting the type to 'float'
    return (X,y) # returning the X , y

def get_theta(costFunction , X , y , iter = 400):

    options = {'maxiter':iter} # maximum number of iterations 

    row , col = X.shape

    initial_theta = np.zeros(col)

    res = optimize.minimize(
        costFunction, 
        initial_theta ,
        (X,y),
        jac=True,
        method='TNC',
        options = options
    )
    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    return ( cost , theta )

def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    
    g = np.zeros(z.shape)

    g = 1 / (1 + np.exp(-z))

    return g

def costFunction(theta, X, y):
    
    m = y.size  # number of training examples

    J = 0
    grad = np.zeros(theta.shape) # 

    h = sigmoid(X.dot(theta.T)) # sigmoid function 
    
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))
    grad = (1 / m) * (h - y).dot(X)
    
    return J, grad

def load_data(url):
    df=pd.read_csv(url,header=None);
    return ( df.iloc[:,:-1] , df.iloc[:,-1])

def run():
    
    X , y = load_data('./marks.txt')

    ones = X[y==1] # features X where y == 1
    zeros = X[y==0] # features X where y == 0

    #X,y = get_data()
    row , col = X.shape
    # Add intercept term to X
    X = np.concatenate([np.ones((row, 1)), X], axis=1)
    


    (cost,theta)=get_theta(costFunction , X , y )

    print('cost => {} , theta => {}'.format(cost,theta) )

    #print(' x ',X[:,1:3]) # prints col 0 , 1 

    # calculate min of X - 2 , max of X + 2 
    x_treme = np.array([ np.min(X[:,1]) - 2 , np.max(X[:,1]) + 2 ]) 

    # calculate y extreme 
    #y_treme = (-1. / theta[2]) * ( theta[1] * x_treme + theta[0] )
    
    y_treme = - (( np.dot(theta[1] ,x_treme) ) + theta[0] ) / theta[2]

    plt.plot(x_treme , y_treme)

    plt.scatter(ones[0],ones[1] , label="1's ")
    plt.scatter(zeros[0],zeros[1], label="0's ")

    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    run()
