import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf 

housing_data = fetch_california_housing()

#print(housing_data.data.shape)

m,n=housing_data.data.shape

housing_data_plus_bias = np.c_[np.ones((m,1)),housing_data.data]

scalar = StandardScaler()

#scalar.fit(housing_data_plus_bias)

scaled_housing_data_plus_bias = scalar.fit_transform(housing_data_plus_bias)

print(scaled_housing_data_plus_bias)

# print(housing_data_plus_bias)

# print(housing_data_plus_bias.shape)

learning_rate=0.01
epochs = 1000


x= tf.constant(housing_data_plus_bias,dtype=tf.float32,name='x')
y= tf.constant(housing_data.target.reshape(-1,1) ,dtype=tf.float32,name='y')
x_t = tf.transpose(x)
#theta = tf.matmul(tf.matrix_inverse(tf.matmul(x_t,x)),tf.matmul(x_t,y))

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
y_pred= tf.matmul(x,theta,name='y_pred')
error = (y_pred - y)
mse=tf.reduce_mean(tf.square(error),name='mse')
gradients = 2/m * tf.matmul(x_t,error)
training_ops = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        if i % 100 == 0:
            print(" MSE = {}".format(mse.eval()))
        
        sess.run(training_ops)
    
    print(" Best Theta {} ".format(theta.eval()))
