import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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

def run():
    X , y = load_data('./marks.txt')
    #X,y = get_data()
    
    ones = X[y==1]
    zeros = X[y==0]

    plt.scatter(ones[0],ones[1] , label="1's ")
    plt.scatter(zeros[0],zeros[1], label="0's ")

    lr = LogisticRegression()
    lr.fit(X,y)

    # x0 * w0 + x1 * w1 + b = 0
    # x1 = - ( ( x0 * w0 ) + b  ) / w1

    w0 , w1 = (lr.coef_[0][0] , lr.coef_[0][1])
    b = lr.intercept_[0]

    print(' w0  = {} , w1 = {} , b = {} '.format(w0 , w1 , b) )

    x_treme = [np.min(X[0]) , np.max(X[1])]
    y_treme = - (( np.dot(w0 ,x_treme) ) + b ) / w1

    print( x_treme , y_treme)

    #print(lr.coef_,lr.intercept_)

    plt.plot(x_treme , y_treme )

    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    run()
