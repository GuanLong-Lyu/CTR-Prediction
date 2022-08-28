from audioop import cross
import tensorflow as tf
from tensorflow.keras.layers import Layer

class crossing_layer(Layer):
    def __init__(self,crossing_num):
        super(crossing_layer, self).__init__()
        self.crossing_num = crossing_num
    
    def build(self, input_shape):
    
    def call(self, inputs):
