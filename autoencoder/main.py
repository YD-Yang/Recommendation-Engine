
import pandas as pd
import numpy as np

import sys 
sys.path.insert(0, 'U:\\autoencoder_CF_package\\data')
sys.path.insert(0, 'U:\\autoencoder_CF_package')

sys.path.remove('U:\\autoencoder_CF_package')
sys.path.remove('U:\\autoencoder_CF_package\\data')


from DAE import DAE
import model_helper
import data 
import train 


from sklearn.model_selection import train_test_split


train_raw, test_raw = train_test_split(df2, test_size = 0.2, random_state = 0)

#data prepare 
from collections import OrderedDict

df_test = test_raw.set_index('id')
df_test = df_test.unstack().swaplevel().to_frame().reset_index()
df_test.columns = ['userID', 'itemID', 'rating']
df_test = df_test.dropna()
df_test.rating = df_test.rating +1

df_train = train_raw.set_index('id')
df_train = df_train.unstack().swaplevel().to_frame().reset_index()
df_train.columns = ['userID', 'itemID', 'rating']
df_train = df_train.dropna()
df_train.rating = df_train.rating + 1

#from folder preprocess_data
#from preprocess_data import get_dataset 
#training_set, test_set = get_dataset(df_train, df_test)


#from tf_record_write save files to train/test folder 
from tf_record_writer import tf_record_writer 
tf_record_writer(df_train, df_test)

from train import main

main(_)
