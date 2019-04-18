import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
import sys
import cv2

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
def main(*args):
    #Step 1 - download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = './data/'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
  
    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
  
    model_fn = 'tensorflow_inception_graph.pb'
    
    #Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    operations=[]
    for operation in graph.get_operations():
        #if operation.type=='Conv2D' and  'import/' in operation.name:
        operations.append(operation)
    print('len , before => ',len(operations))        
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    operations = tf.import_graph_def(graph_def, {'input':t_preprocessed})
    
    operations=[]
    for operation in graph.get_operations():
        #if operation.type=='Conv2D' and  'import/' in operation.name:
        operations.append(operation)
        #print('operation =>',operation)
    
    print('len , after => ',len(operations))
    writer = tf.summary.FileWriter('logs', sess.graph)
    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
    
    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))
  
 #####HELPER FUNCTIONS. I didn't go over these in the video for times sake. They are mostly just formatting functions. Scroll 
 #to the bottom #########################################################################################################
 ########################################################################################################################
 ############################################################
 
    # Helper functions for TF Graph visualization
    #pylint: disable=unused-variable
    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def
      
    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def
      
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()
        
    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5
    
    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)
    
    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
            #writer.add_summary(g)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
        showarray(visstd(img))
        
    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            #print('wrap =>',out)
            def wrapper(*args, **kw):
                for w in args:
                    #print('wrapper => ',type(w))
                    return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap
    
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        print('resize ',img,size)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)
    #print('resize',resize)
    
    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        img_summaries = []
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                #print('calc_grad ',g.shape,g.__class__)
                #img_sum = tf.Summary.Image(encoded_image_string=img,
                #                 height=img.shape[0],
                #                 width=img.shape[1])
                #img_summaries.append(tf.Summary.Value(tag='%d' % (i), image=img_sum))
                #writer.add_summary(summary)
                grad[y:y+sz,x:x+sz] = g
        #summary = tf.Summary(value=img_summaries)
        #writer.add_summary(summary)
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)    

    #BACK TO CODE IN THE VIDEO###########################################################################################
    ########################################################################################################
    ##############################################################################
    
    #CHALLENGE - Write a function that outputs a deep dream video
    #def render_deepdreamvideo():
        
    def save_image(input_file,output_file):
        print(' Applying Deepdream to __ ',input_file)
        img = PIL.Image.open(input_file)
        img = np.float32(img)
        
        array = render_deepdream(tf.square(T('mixed4c')), img)
        
        image = PIL.Image.fromarray(array)
        image.save(output_file)
        print('Deep Dream applied to __ ',output_file)

    def save_to_video(input_file,output_file):
        video = cv2.VideoCapture(input_file)
        print(' Applying Deepdream to __ ',input_file)
        video_writter = None
        
        index = 0

        if video.isOpened() is False:
            print('Unable to open __ input_file')

        while(video.isOpened()):
            ret , frame = video.read()

            if frame is None:
                break
            
            # Apply Gradients to the Frame
            output_frame = render_deepdream(tf.square(T('mixed4c')), frame) 

            #Initiallzing the video_writter @ the beginning
            if video_writter is None:
                frame_size = ( output_frame.shape[0] , output_frame.shape[1] )
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writter = cv2.VideoWriter(output_file,fourcc, 20.0, frame_size)
            
            index+=1
            video_writter.write(output_frame)
            print('Printing __ Frame : ',index)
        
        print('Applied Deepdream to __ ',input_file)
        cap.release()
        video_writter.release()
        cv2.destroyAllWindows()

    def render_deepdream(t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        #print('t_score => ',t_score)
        #print('t_grad => ',t_grad)
        # split the image into a number of octaves
        img = img0
        
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            '''
            print('resize => ',img.shape,lo.shape,hw)
            print('hw => ',np.float32(hw)) 
            print('hw/octave_scale => ',np.float32(hw)/octave_scale )
            print('np.int32 ', np.int32(np.float32(hw)/octave_scale) )
            '''
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)
        
        #for octave in octaves:
        #    print('octaves => ',octave.shape)
            '''
            120 , 162
            86 , 115
            61 , 82
            '''
        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
            
            #this will usually be like 3 or 4 octaves
            #Step 5 output deep dream image via matplotlib
            #showarray(img/255.0)
            array = (img/255.0)
            output = np.uint8(np.clip(array, 0, 1)*255)
            return output
            
         
  
   	#Step 3 - Pick a layer to enhance our image
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139 # picking some feature channel to visualize
    
    #open image
    img0 = PIL.Image.open('pilatus8000.jpg')
    img0 = np.float32(img0)

    if args[1] == '--image':
        save_image(args[0],args[2])
    elif args[1] == '--video':
        save_to_video(args[0],args[2])
    else:
        print('EXIT 0')
    #Step 4 - Apply gradient ascent to that layer
    #render_deepdream(tf.square(T('mixed4c')), img0)
      
# demo.py <input.mp4> --video <output.mp4>(optional)
# demo.py <input.jpg> --image <output.jpg>(optional)

if __name__ == '__main__':
    if (len(sys.argv) < 3):        print( ' Arguments required ...  ')
        print( ' demo.py <input.mp4> --video <output.mp4>(optional) ')
        print( ' demo.py <input.jpg> --image <output.jpg>(optional) ')
    else:
        input_file = sys.argv[1]
        FLAG = sys.argv[2]

        output_file = None
        
        if len(sys.argv) == 4:
            output_file = sys.argv[3]
        else:
            output_file = ('output.jpg','output.mp4')[FLAG == '--video']
        
        main(input_file,FLAG,output_file)

