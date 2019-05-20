import os
import tensorflow as tf
#import keras.layers.max_pooling2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2

def write_color_image(path,count):
    image=cv2.imread(path)
    image=cv2.resize(image,(128,128))
    cv2.imwrite("./training/colored/colored_{}.jpg".format(count),image)

def write_grey_image(path,count):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(128,128))
    cv2.imwrite("./training/grey/grey_{}.jpg".format(count),image)


def create_images_to_train(count=0):
    folders = os.listdir('./flower_photos')
    print('folders',folders)
    for folder in folders:
        paths = os.listdir('./flower_photos/'+folder)
        
        for path in paths:
            image_path = "./flower_photos/{}/{}".format(folder,path)
            
            write_color_image(image_path,count)
            write_grey_image(image_path,count)
            count+=1

def get_datasets():
    img_list=[]
    colored = os.listdir('./training/colored')
    for image in colored:
        img = cv2.imread("./training/colored/{}".format(image))
        img_list.append(np.array(img))

    source = np.asarray(img_list)
    img_list=[]

    grey = os.listdir('./training/grey')
    for image in grey:
        img = cv2.imread("./training/grey/{}".format(image),0) # 0 for greyscale
        img_list.append(np.array(img))

    target = np.asarray(img_list)
    target = target[:, :, :, np.newaxis]

    return (source,target)

def run():
    #create_images_to_train()
    (source,target) = get_datasets()
    print(source.shape)
    print(target.shape)
    
    #Input Layers
    inputs = tf.placeholder(tf.float32,(None,128,128,3),name='Input_of_Autoencoder')
    targets= tf.placeholder(tf.float32,(None,128,128,1))

    # Encoder Layers
    #network = tf.keras.layers.Conv2D(128, 2, activation = tf.nn.relu)(inputs)
    network =tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu)
    network =tf.layers.max_pooling2d(network, 2, 2, padding = 'same')

    # Decoder Layers    
    network = tf.image.resize_nearest_neighbor(network, tf.constant([129, 129]))
    output =  tf.layers.conv2d(network, 1, 2, activation = None, name = 'outputOfAuto')
    #Loss
    loss = tf.reduce_mean(tf.square(output-targets))

    #Optimization
    train_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.initialize_all_variables()

    batch_size = 32
    epoch_num = 50
    
    save_to_path = './model/AEColor_To_Gray.ckpt'
    
    saver = tf.train.Saver(max_to_keep = 3)
    
    batch_img = source[0:batch_size]
    batch_out = target[0:batch_size]

    num_of_images = 1531
    
    num_batches = num_of_images//batch_size
    sess = tf.Session()
    sess.run(init)
    for ep in range(epoch_num):
        batch_size = 0
        for batch_n in range(num_batches):
            _, c = sess.run([train_opt, loss], feed_dict = {inputs: batch_img, targets: batch_out})
            print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))
            batch_img = source[batch_size: batch_size+32]
            batch_out = target[batch_size: batch_size+32]
            batch_size += 32
            saver.save(sess, save_to_path, global_step = ep)
            
    sess.close()

    def test():
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver.restore(sess,  './model/AEColor_To_Gray.ckpt-49')
        test_data = []
        
        filenames = os.listdir("./flower_images/flower_images")
        filenames = ["./flower_images/flower_images/{}".format(file) for file in filenames]
        
        for file in filenames[0:100]:
            test_data.append(np.array(cv2.imread(file)))
        test_dataset = np.asarray(test_data)
        print(test_dataset.shape)
        batch_imgs = test_dataset
        gray_imgs = sess.run(output, feed_dict = {inputs: batch_imgs})
    
        for i in range(gray_imgs.shape[0]):
            cv2.imwrite('./generated/gen_gray_' +str(i) +'.jpeg', gray_imgs[i])

    test()

if(__name__== "__main__"):
    run()

