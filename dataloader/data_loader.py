import tensorflow as tf
from tensorflow.keras.layers import Embedding,Input,Concatenate,Flatten
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder

# preprocessing of dense features
# 对连续数据填补空值
def process_feats(data, feats, dense=True, train=True, scaler=None, dict_label_encoder= None):
    """Process dense or sparse features

    Fill the NaN values, transformations.

    Args: 
        data: A Pandas DataFrame contains the dense features. 
        feats: Dense features' name in DataFrame.

    Returns:
        A Dataframe after fill the NaN values and transformations.  
    """
    
    if dense:
        # fill NaN values for Dense features
        data_copy = data.copy()
        data_copy = data_copy[feats].fillna(0) # fill NaN for Dense features as intger 0
        
        if train:
            scaler = StandardScaler()
            data_copy[feats] = scaler.fit_transform(data_copy[feats]) # standardscaler fit_transform 
        else:
            data_copy[feats] = scaler.transform(data_copy[feats]) # train set's standardscler transform only.
        
        return data_copy,scaler

        
    else: 
        # fill NaNN values for Sparse features
        data_copy = data.copy()
        data_copy = data_copy[feats].fillna("-1") # fill NaN for Sparse features as "-1"
        

        if train:
            dict_label_encoder = dict()
            for feat in feats:
                label_encoder = LabelEncoder()
                data_copy[feat] = label_encoder.fit_transform(data_copy[feat])
                dict_label_encoder[feat] = label_encoder
        else:
            for feat in feats:
                label_encoder = dict_label_encoder[feat]
                data_copy[feat] = label_encoder.transform(data_copy[feat])

        return data_copy,dict_label_encoder


# Get Dense Input
def get_dense_input(feats):
    dense_inputs = []
    for feat in feats:
        dense_input = Input(shape=(1,),name=feat)
        dense_inputs.append(dense_input)
    dense_inputs = Concatenate(axis=1)(dense_inputs)
    return dense_inputs

# Get Sparse Innput
def get_sparse_input(data,feats,embedding_size = 8):
    sparse_inputs = []
    for feat in feats:
        sparse_input = Input(shape=(1,),name=feat)
        sparse_inputs.append(sparse_input)
    
    sparse_embeddings = []
    for i,sparse_input in enumerate(sparse_inputs):
        f = feats[i]
        voc_size = data[f].nunique()
        reg = tf.keras.regularizers.l2(0.7)
        embed = Embedding(
            voc_size+1,
            embedding_size,
            embeddings_regularizer=reg,

        )(sparse_input)
        embed = Flatten()(embed)
        sparse_embeddings.append(embed)
    sparse_embeddings = Concatenate(axis=1)(sparse_embeddings)
    return sparse_embeddings

def get_model_input(dense_input,sparse_input):
    model_input = Concatenate(axis=1)([dense_input,sparse_input])

    return model_input
