from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def compute_error(data,m,b):
        total_error=0
        N=float(len(data))
        for i in range(len(data)):
                [x,y]=data[i]
                y_ = (m * x ) + b
                total_error += (y-y_)** 2
        return ((1/N)*total_error) 

'''
error function :
y_=(m*x)+b
mean_squared_loss = (1/N)((y-y_))**2

partial deriatives of error function :
w_ = -(2/N)(x*(y-(m*x+b))) 
b_ = -(2/N)(y-(m*x+b))
'''

def gradient_descent(data,m,b,learning_rate=0.001):
        N=float(len(data))

def step_gradient_descent(data,m,b,learning_rate=0.0001):
        b_gradient= m_gradient = 0
        N=float(len(data))
        for i in range(len(data)):
                [x,y]=data[i]
                y_=(m*x)+b
                m_gradient+= - (2/N)*(x*(y-y_))
                b_gradient+= -(2/N)*(y-y_)
        #print("m ={}, b ={}".format(m_gradient,b_gradient))
        m_new = m-(learning_rate*m_gradient)
        b_new = b-(learning_rate*b_gradient)
        return (m_new,b_new)

def plot(data,m , b):
        x=np.array(data[:,0])
        y=np.array(data[:,1])
        m=float(format(m,'.4g')[:3])
        b=float(format(b,'.4g')[:3])
        #print(format(m,'.2g'),float(format(b,'.4g')[:4]))
        print(m,b)
        y_=[ b+(m*y_s) for y_s in y]
        y_ = (m*x)+b
        print(y_)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        #print(data[0],data[1])
        #print(data[:,0],data[:,1])
        ax.scatter(x,y,c='red',label="points (x,y)")
        ax.plot(x,y_,label="line of best fit")
        ax.set_xlabel('X (cycled)')
        ax.set_ylabel('Y (cal burned)')

        plt.title('Regression Line')
        plt.show()

def plot_3d(error,m,b):
        fig=plt.figure()
        #ax = fig.gca(projection='3d')
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(m,b,error,c="r",label='valley')
        
        ax.set_xlabel('weight')
        ax.set_ylabel('bias')
        ax.set_zlabel('Error(w,b)')

        ax.legend()
        plt.title('Finding Minima')
        plt.show()


def perform_gradient_descent(data,m,b,epochs=1000):
        m_array=b_array=[]
        for i in range(epochs):
                if(i % 100 == 0 ):
                        print("Running {}/{}".format(i,epochs))
                (m,b) = step_gradient_descent(data,m,b)
                m_array.append(m)
                b_array.append(b)
        #np.array(m_array),np.array(b_array))

        plot(data,m,b) 

        return (m,b,m_array,b_array)
                
def run():
        print('Running')
        data = genfromtxt('data.csv',delimiter=',')
        initial_m = 0
        initial_b = 0
        error=compute_error(data,initial_m,initial_b)
        print("Error @ inital stage : {}".format(error))
        (m,b,m_array,b_array)=perform_gradient_descent(data,initial_m,initial_b)
        #print('gradient ',m,b)
        N=len(m_array)
        e_array=[]

        for i in range(N):
                e_array.append(compute_error(data,m_array[i],b_array[i]))
        
        #plot_3d(e_array,m_array,b_array) function for finding Minima

        error=compute_error(data,m,b)
        print("Error after Performing Gradient Descent : {}".format(error))
        
if __name__ == '__main__':
    run()

