import tensorflow as tf
import cv2
import numpy as np

import os
import sys
import argparse
import pdb

class yolo_tiny_model:
    model_input = None
    mode = None
    disp_console = None
    model_output = None
    yolo_variables = None

    sess = None
    weights_file = 'weights/YOLO_tiny.ckpt'
    
    alpha = 0.1
    
    
    def __init__(self, init_dict):
        self.model_input = init_dict['yolo_model_input']
        self.mode = init_dict['yolo_mode']
        self.disp_console = init_dict['yolo_disp_console']
        self.sess = init_dict['session']

        self.model_output = self.build_graph(self.model_input, self.mode)
        
        self.yolo_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1:]
        # unused, but you can check name like this
        YOLO_variables_name = [variable.name for variable in self.yolo_variables]
        


    def build_graph(self, image, mode):
        assert mode=="init_model" or mode=="reuse_model"
        self.conv_1 = self.conv_layer(1,image,16,3,1,'Variable:0', 'Variable_1:0',mode=mode)
        self.pool_2 = self.pooling_layer(2,self.conv_1,2,2,mode=mode)
        self.conv_3 = self.conv_layer(3,self.pool_2,32,3,1,'Variable_2:0', 'Variable_3:0',mode=mode)
        self.pool_4 = self.pooling_layer(4,self.conv_3,2,2,mode=mode)
        self.conv_5 = self.conv_layer(5,self.pool_4,64,3,1,'Variable_4:0', 'Variable_5:0',mode=mode)
        self.pool_6 = self.pooling_layer(6,self.conv_5,2,2,mode=mode)
        self.conv_7 = self.conv_layer(7,self.pool_6,128,3,1,'Variable_6:0', 'Variable_7:0',mode=mode)
        self.pool_8 = self.pooling_layer(8,self.conv_7,2,2,mode=mode)
        self.conv_9 = self.conv_layer(9,self.pool_8,256,3,1,'Variable_8:0', 'Variable_9:0',mode=mode)
        self.pool_10 = self.pooling_layer(10,self.conv_9,2,2,mode=mode)
        self.conv_11 = self.conv_layer(11,self.pool_10,512,3,1,'Variable_10:0', 'Variable_11:0',mode=mode)
        self.pool_12 = self.pooling_layer(12,self.conv_11,2,2,mode=mode)
        self.conv_13 = self.conv_layer(13,self.pool_12,1024,3,1,'Variable_12:0', 'Variable_13:0',mode=mode)
        self.conv_14 = self.conv_layer(14,self.conv_13,1024,3,1,'Variable_14:0', 'Variable_15:0',mode=mode)
        self.conv_15 = self.conv_layer(15,self.conv_14,1024,3,1,'Variable_16:0', 'Variable_17:0',mode=mode)
        self.fc_16 = self.fc_layer(16,self.conv_15,256,'Variable_18:0', 'Variable_19:0',flat=True,linear=False,mode=mode)
        self.fc_17 = self.fc_layer(17,self.fc_16,4096,'Variable_20:0', 'Variable_21:0',flat=False,linear=False,mode=mode)
        #skip dropout_18
        self.fc_19 = self.fc_layer(19,self.fc_17,1470,'Variable_22:0', 'Variable_23:0',flat=False,linear=True,mode=mode)
        self.c = tf.reshape(tf.slice(self.fc_19,[0,0],[1,980]),(7,7,20))
        self.s = tf.reshape(tf.slice(self.fc_19,[0,980],[1,98]),(7,7,2))
        #self.probs = tf.Variable(tf.ones(shape=[]))
        #self.probs = tf.placeholder('float32',[None,7,7,2])
        #self.com=tf.constant(0.2*np.ones(98,dtype='float32'))
        self.p1 = tf.multiply(self.c[:,:,14],self.s[:,:,0])
        self.p2 = tf.multiply(self.c[:,:,14],self.s[:,:,1])
        self.p = tf.stack([self.p1,self.p2],axis=0)
        #for i in range(2):
        #self.probs[:,:,i].assign(tf.multiply(self.c[:,:,14],self.s[:,:,i]))
        #self.probs=tf.concat([self.p1,self.p2],0)
        #self.yan=tf.reduce_sum(tf.maximum(self.probs,0.2))
        #self.yan=tf.reduce_sum(self.probs)
        Cp = tf.reduce_max(self.p) # confidence for people
        
        return Cp
    
    def init_variables(self, YOLO_variables):
        init = tf.global_variables_initializer()
        # init = tf.variables_initializer(YOLO_variables)
        self.sess.run(init)

    def load_weight(self, variables):
        saver = tf.train.Saver(variables)#[0:-1][1:-4]
        saver.restore(self.sess,self.weights_file)
        
    def set_output_tensor(self):
        self.model_output = None
        
    def get_output_tensor(self):
        
        return self.model_output
    
    def get_yolo_variables(self):
        
        return self.yolo_variables
    
    def get_model_sess(self):
        
        return self.sess
    
    # release related resources
    def terminate_model(self):
        self.sess.close()
        
    def conv_layer(self, 
                   idx, 
                   inputs, 
                   filters, 
                   size, 
                   stride, 
                   weight_name, 
                   biases_name, 
                   mode="init_model"):

        channels = inputs.get_shape()[3]
        if mode=="init_model":
            # weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1))
            weight = tf.get_variable(name=weight_name[:-2],shape=[size,size,int(channels),filters],dtype=tf.float32)
            # biases = tf.Variable(tf.constant(0.1, shape=[filters]))
            biases = tf.get_variable(name=biases_name[:-2],shape=[filters],dtype=tf.float32)
        if mode=="reuse_model":
            weight = tf.get_variable(name=weight_name[:-2])
            biases = tf.get_variable(name=biases_name[:-2])

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')    
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')    
        if self.disp_console and mode=="init_model": print('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx,size,size,stride,filters,int(channels)))

        return tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')

    def pooling_layer(self,
                      idx,
                      inputs,
                      size,
                      stride,
                      mode="init_model"):

        if self.disp_console and mode=="init_model": print('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx,size,size,stride))

        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

    def fc_layer(self,
                 idx,
                 inputs,
                 hiddens,
                 weight_name,
                 biases_name,
                 flat = False,
                 linear = False,
                 mode="init_model"):

        input_shape = inputs.get_shape().as_list()        
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_transposed = tf.transpose(inputs,(0,3,1,2))
            inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        
        if mode=="init_model":
            # weight = tf.Variable(tf.truncated_normal([dim,hiddens], stddev=0.1))
            weight = tf.get_variable(name=weight_name[:-2],shape=[dim,hiddens],dtype=tf.float32)
            # biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))
            biases = tf.get_variable(name=biases_name[:-2],shape=[hiddens],dtype=tf.float32)
        if mode=="reuse_model":
            weight = tf.get_variable(name=weight_name[:-2])
            biases = tf.get_variable(name=biases_name[:-2])
        
        if self.disp_console and mode=="init_model": print('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx,hiddens,int(dim),int(flat),1-int(linear))    )
        if linear : return tf.add(tf.matmul(inputs_processed,weight),biases,name=str(idx)+'_fc')
        ip = tf.add(tf.matmul(inputs_processed,weight),biases)
        return tf.maximum(self.alpha*ip,ip,name=str(idx)+'_fc')

# define a callback for argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# arguments parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_folder_dir', type=str, help='Dir with videos to be processed.')
    parser.add_argument('output_folder_dir', type=str, help='Dir for processed videos to save.')
    parser.add_argument('mode', type=str, help='Processing mdoe for adversary attack.')
    
    parser.add_argument('--useEOT', type=str,
                        help='Create EOT attack graph instead of single angle graph.', default=44)
    parser.add_argument('--disp_console', type=str2bool,
                        help='Create EOT attack graph instead of single angle graph.', default=True)

    return parser.parse_args(argv)

# this main() is a demo
def main(args):
    tf_input = tf.placeholder('float32', [1,448,448,3])
    init_dict = {'yolo_model_input': tf_input,
                 'yolo_mode': args.mode,
                 'yolo_disp_console': args.disp_console,
                 'session': tf.Session()}
    
    yolo_detector = yolo_tiny_model(init_dict)
    yolo_variables = yolo_detector.get_yolo_variables()
    
    yolo_detector.init_variables(yolo_variables)
    yolo_detector.load_weight(yolo_variables)

    output = yolo_detector.get_output_tensor()
    
    pic = cv2.imread('test/任务.jpeg')
    pic = cv2.resize(pic,(448,448))/255*2-1
    feed_batch = pic[np.newaxis, :]
    feed_dict = {tf_input: feed_batch}
    
    # 必须要获得model的会话才可以操纵她
    sess = yolo_detector.get_model_sess()
    result = sess.run([output], feed_dict=feed_dict)
    print(result)
    yolo_detector.terminate_model()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))