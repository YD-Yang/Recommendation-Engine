# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:09:36 2018

@author: YXY3512
https://github.com/ShaunZia/MovieRec-CollabFilt-RBM/blob/master/MovieRec-CollabFilt-RBM.ipynb
"""

import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
################

df2 = pd.read_csv("U:/Projects/Recomendation Egngine/HEDIS/raw data/mbr_channel_python.csv")
#df2 = df2.set_index('sdr_person_id')
df2.columns = [x.lower() for x in df2.columns]
c=list(df2.columns.values)

from collections import OrderedDict

df = df2.set_index('sdr_person_id')

df = df.unstack().swaplevel().to_frame().reset_index()
df.columns = ['userID', 'itemID', 'rating']
df = df.dropna()

""" train test split by users 
user_ls =list( df.userID.unique())
user_train, user_test = train_test_split(user_ls, test_size = .2, random_state = 10 )
train_df = df[df['userID'].isin( user_train)]
test_df = df[df['userID'].isin( user_test)]

"""

item_feature = pd.read_csv("U:/Projects/Recomendation Egngine/HEDIS/raw data/channel_test.csv")
item_feature['list_index'] = item_feature.index

merge_df = item_feature.merge(df, on = 'itemID')
#add a small number to zero rating 
merge_df.rating = merge_df.rating.map(lambda x: x + 0.001 if x < 0.001 else x)


#group by userID 
userGroup = merge_df.groupby('userID')


#Amount of users used for training
train, test = train_test_split(userGroup, test_size = 0.2, random_state = 10 )
amountOfUsedUsers = 630000
#Creating the training list
trX = []
#For each user in the group
for userID, curUser in userGroup:
    #Create a temp that stores every item's rating
    temp = [0]*len(item_feature)
    #For each item in curUser's item list
    for num, item in curUser.iterrows():
        #Divide the rating by 5 and store it
        temp[item['list_index']] = item['rating']
    #Now add the list of ratings into the training list
    trX.append(temp)
    #Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1


###model parameters
hiddenUnits = 20
visibleUnits = len(item_feature)
vb = tf.placeholder("float", [visibleUnits]) #Number of unique items
hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
W = tf.placeholder("float", [visibleUnits, hiddenUnits])


#Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0= tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
#Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

#Learning rate
alpha = 0.5
#Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = v0 - v1
err_sum = tf.reduce_mean(err * err)



#Current weight#Current 
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)
#Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)
#Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)
#Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())



epochs = 20
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_nb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_nb}))
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()





#######################################################################

#Selecting the input user
inputUser = [trX[50]]


#Feeding in the user and reconstructing the input#Feeding 
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={ v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})

scored_df_samp= item_feature
scored_df_samp["Recommendation Score"] = rec[0]
print(scored_df_samp.sort_values(["Recommendation Score"], ascending=False).head(20))



""" Recommend User items he has not received yet """

# Find the mock user's UserID from the data
print(merge_df.iloc[50])  # Result you get is UserID 1000122138

# Find all item the mock user has  before
use_df_samp= merge_df[merge_df['userID'] == 1000122138]
print(use_df_samp.head())

""" Merge all movies that our mock users has watched with predicted scores based on his historical data: """

# Merging movies_df with ratings_df by MovieID
merged_df_samp = scored_df_samp.merge(use_df_samp, on='itemID', how='outer')

# Dropping unnecessary columns
merged_df_samp = merged_df_samp.drop('list_index_y', axis=1).drop('userID', axis=1)

# Sort and take a look at first 20 rows
print(merged_df_samp.sort_values(['Recommendation Score'], ascending=False).head(20))


#######################################################################
#######################################################################


