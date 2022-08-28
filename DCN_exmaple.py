import pandas as pd
from dataloader.data_loader import process_feats,get_dense_input,get_sparse_input,get_model_input

# read csv file 
df_criteo_train = pd.read_csv("CTR-Prediction/dataset/criteo_small/criteo_small_train.csv")
df_criteo_test = pd.read_csv("CTR-Prediction/dataset/criteo_small/criteo_small_test.csv")

# get sparse,dense feature names
sparse_cols = [fea for fea in df_criteo_train.iloc[:,2:].columns if df_criteo_train[fea].dtypes == "object"]
dense_cols = [fea for fea in df_criteo_train.iloc[:,2:].columns if df_criteo_train[fea].dtypes != "object"]
label_col = ["0"]

# get data
train_data_sparse, dict_label_encoder = process_feats(data= pd.concat([df_criteo_train,df_criteo_test],axis=0),feats = sparse_cols, dense= False,train=True)
train_data_dense, scaler = process_feats(data=df_criteo_train,feats=dense_cols, dense= True, train= True)

test_data_sparse, dict_label_encoder = process_feats(data= df_criteo_test,feats = sparse_cols, dense= False,train=False,dict_label_encoder=dict_label_encoder)
test_data_dense, scaler = process_feats(data=df_criteo_test,feats=dense_cols, dense= True, train=False, scaler=scaler)

# get input
# get dense input
dense_inputs = get_dense_input(dense_cols)
print(dense_inputs)

# get sparse embedding input
sparse_inputs = get_sparse_input(df_criteo_train,sparse_cols)
print(sparse_inputs)

# get model's input, by concatenate dense_input and sparse_input
model_input = get_model_input(dense_input=dense_inputs,sparse_input=sparse_inputs)
print(model_input)