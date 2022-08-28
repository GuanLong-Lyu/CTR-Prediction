from msilib.schema import SelfReg
import tensorflow as tf
from tensorflow.keras.layers import Layer

class crossing_layer(Layer):
    def __init__(self,crossing_num,reg = 0.2):
        super(crossing_layer, self).__init__()
        self.crossing_num = crossing_num
        self.reg = reg
    
    def build(self, input_shape):
        dim = input_shape[-1]
        self.kernals = [self.add_weight(
            name = "kernal"+str(i),
            shape = (dim,1), # the parameter is vector in Deep Cross network first version
            initializer = tf.keras.initializers.GlorotNormal(),
            regularizer = tf.keras.regularizers.l2(0.1),
            trainable = True) for i in range(self.crossing_num)]
        
        self.bias = [self.add_weight(
            name = "bias" + str(i),
            shape = (dim,1),
            initializer = tf.keras.initializers.Zeros(),
            trainable = True) for i in range(self.crossing_num)]
        super(crossing_layer,self).build(input_shape)

    def call(self, inputs):
        x0 = 
        xl = 

