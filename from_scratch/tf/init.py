import tensorflow as tf

print('Using tensorflow ',tf.__version__)

x = tf.Variable(5,name="X")
y = tf.Variable(7,name="Y")
f=(x*x*y) + y +2
f2=(x*x*y)
result =None

'''
Method 0
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
'''

'''
Method 2
with tf.Session() as sess:
    x.initializer.run() #tf.get_default_session.run(x.initializer)
    y.initializer.run()
    result=f.eval()     #tf.get_default_session.run(f)
 
'''

'''
Method 3
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run() # handles all the initialisation of x , y
    result = f.eval()

'''

'''
Method 4

'''

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result1,result2 = sess.run([f,f2]) #(x*x*y) will be reused
sess.close()


x1=  tf.Variable(2)
same_graph = x1.graph is tf.get_default_graph()

print(same_graph)

x2=None
graph_1 = tf.Graph()
with graph_1.as_default():
    x2=tf.Variable(2)

same_graph  = x2.graph is tf.get_default_graph()

print(same_graph)

same_graph  = x2.graph is graph_1

print(same_graph)

print(result1)
print(result2)
