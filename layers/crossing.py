import tensorflow as tf
from tensorflow.keras.layers import Layer

class crossing_layer(Layer):
    def __init__(self,crossing_num,l1_reg_num=0.0,l2_reg_num = 0.0,parameter_type="vector"):
        super(crossing_layer, self).__init__()
        if parameter_type not in ["matrix","vector"]:
            raise ValueError("Parameter type must be matrix or vector")

        self.crossing_num = crossing_num
        self.l1_reg_num = l1_reg_num
        self.l2_reg_num = l2_reg_num
        self.parameter_type = parameter_type
    
    def build(self, input_shape):
        dim = input_shape[-1]
        if self.parameter_type == "vector":
            self.kernel = [
                self.add_weight(
                    name = "kernel"+str(i),
                    shape = [dim,1], # the parameter is a (dim,1) vector 
                    initializer = tf.keras.initializers.GlorotNormal(),
                    regularizer = tf.keras.regularizers.l2(self.l2_reg_num),
                    trainable = True
                ) for i in range(self.crossing_num)
            ]
        elif self.parameter_type == "matrix":
            self.kernel = [
                self.add_weight(
                    name = "kernel"+str(i),
                    shape = [dim,dim], # the parameter is a  (dim,dim) matrix
                    initializer = tf.keras.initializers.GlorotNormal(),
                    regularizer = tf.keras.regularizers.l2(self.l2_reg_num),
                    trainable = True
                ) for i in range(self.crossing_num)
            ]
        
        self.bias = [self.add_weight(
            name = "bias" + str(i),
            shape = [dim,1],
            initializer = tf.keras.initializers.Zeros(),
            trainable = True) for i in range(self.crossing_num)]
        super(crossing_layer,self).build(input_shape)

    def call(self, inputs):
        x0 = inputs
        xl = x0
        if self.parameter_type == "vector":
            print("paramter type: vector")
            for i in range(self.crossing_num):
                dot = tf.matmul(xl,self.kernel[i])
                xl_w = tf.multiply(x0,dot)
                output = tf.nn.bias_add(xl_w,self.bias[i]) + x0
        
        elif self.parameter_type == "matrix":
            print("parameter type: matrix")
            for in range(self.crossing_num):
                
            
        return output
