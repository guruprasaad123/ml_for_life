import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import make_moons

# to get Random N rows X , Y batches
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons[:-test_size]
X_test = X_moons[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

# Creating the One Hot Encoder 
oneHot = OneHotEncoder() 
  
# Encoding 

oneHot.fit(X_moons)
X_moons_hot = oneHot.transform(X_moons).toarray()

oneHot.fit(y_moons_column_vector)
y_moons_hot = oneHot.transform(y_moons_column_vector).toarray()

X_train_hot = X_moons_hot[:-test_size]
X_test_hot = X_moons_hot[-test_size:]
y_train_hot = y_moons_hot[:-test_size]
y_test_hot = y_moons_hot[-test_size:]

print( X_train.shape , X_test.shape , y_train.shape , y_test.shape )
print( X_train_hot.shape , X_test_hot.shape , y_train_hot.shape , y_test.shape)
print( X_moons_hot.shape , y_moons_hot.shape)

m , n = X_train_hot.shape
print(m,n)

learning_rate=0.01

logdir = log_dir('logs') 
# There are n columns in the feature matrix 
# after One Hot Encoding. 
X = tf.placeholder(tf.float32, [None, n]) 
  
# Since this is a binary classification problem, 
# Y can take only 2 values. 
Y = tf.placeholder(tf.float32, [None, 2]) 
  
# Trainable Variable Weights 
W = tf.Variable(tf.zeros([n, 2])) 
  
# Trainable Variable Bias 
b = tf.Variable(tf.zeros([2])) 

# Hypothesis 
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b)) 
  
# Sigmoid Cross Entropy Cost Function 
cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y) )

# Cost summary - 
cost_summary = tf.summary.scalar('cost_summary',cost)

# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost) 
  
# Global Variables Initializer 
init = tf.global_variables_initializer() 

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph() )


n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_hot , y_train_hot, batch_size)
            sess.run(optimizer, feed_dict={X: X_batch, Y: y_batch})
       
        correct_prediction = tf.equal( tf.argmax(Y_hat, 1) , tf.argmax(Y, 1) ) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy_summary',accuracy) 
        
        loss_val , loss_summary , accuracy_val , acc_summary = sess.run([cost,cost_summary,accuracy , accuracy_summary] , { X: X_test_hot, Y: y_test_hot } )
        # loss_val = cost.eval({X: X_test_hot, Y: y_test_hot})
        # sum_loss = sum(sum(loss_val))
        step = epoch * n_batches + batch_index
        
        file_writer.add_summary(loss_summary, step)
        # file_writer.add_summary(acc_summary, step)
        
    
        
        if epoch % 100 == 0:
            
            print("Epoch:", epoch, "\tLoss:", loss_val , "\tAccuracy:",accuracy_val * 100 )

    y_proba_val = Y_hat.eval(feed_dict={X: X_test_hot, Y: y_test_hot})

file_writer.flush()
file_writer.close()