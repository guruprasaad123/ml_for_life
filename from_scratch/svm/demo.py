
#To help us perform math operations
import pandas as pd
import numpy as np
#to plot our data and model visually
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs 
  
def svm_sgd_plot(X, Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 1
    #how many iterations to train for
    epochs = 10000
    #store misclassifications so we can plot how they change over time
    errors = []

    #training part, gradient descent part
    for epoch in range(1,epochs):
        error = 0
        if(epoch%1000 == 0):
            print('Running {} {}'.format(epoch,w))
        for i, x in enumerate(X):
            #misclassification

            if (Y[i]*np.dot(X[i], w)+0.1) < 1:
                #misclassified update for ours weights
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                #correct classification, update our weights
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)
        

    #lets plot the rate of classification errors during training for our SVM
    # plt.plot(errors, '|')
    # plt.ylim(0.5,1.5)
    # plt.axes().set_yticklabels([])
    # plt.xlabel('Epoch')
    # plt.ylabel('Misclassified')
    # plt.show()
    
    return w

def run():
    X, Y = make_blobs(n_samples=100, centers=2, 
                  random_state=0, cluster_std=0.40) 
  


    w = svm_sgd_plot(X,Y)

    # plotting scatters  
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring'); 
    x2=[w[0],w[1],w[1],w[0]]
    x3=[w[0],-w[1],w[1],w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.show()

if __name__ == "__main__":
    run()

