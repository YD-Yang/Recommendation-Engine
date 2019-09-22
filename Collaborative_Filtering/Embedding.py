


import pandas as pd
import numpy as np
import os

import surprise
import pandas as pd
import numpy as np
import datetime
from surprise.model_selection import GridSearchCV
from scipy import sparse


###########################################################################################################################
# continuous IDs
def proc_col(col):
    uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    idx2name = {i: e for i, e in enumerate(name2idx.keys())}
    return idx2name, np.array([name2idx[x] for x in col]), len(uniq)


def encode_data(df):
    idx2user, user_col, num_users = proc_col(df.userID)
    idx2item, item_col, num_item = proc_col(df.itemID)
    df.userID = user_col
    df.itemID = item_col
    return df, idx2user, idx2item, num_users, num_item

df, idx2user, idx2item, num_users, num_item = encode_data(df)

#create embedding 

def create_embedings(n, num_factors):
    embedding = 6 * np.random.random((n, num_factors)) / num_factors
    return embedding

def df2matrix(df, nrows, ncols, column_name="rating"):
    values = df[column_name].values
    ind_item = df['itemID'].values
    ind_user = df['userID'].values
    return sparse.csc_matrix((values, (ind_user, ind_item)), shape=(nrows, ncols))

Y = df2matrix(df, num_users, num_item)

def sparse_multiply(df, emb_user, emb_item):
    df["prediction"] = np.sum(emb_user[df["userID"].values] * emb_item[df["itemID"].values],axis=1)
    return df2matrix(df, emb_user.shape[0], emb_item.shape[0], column_name="prediction")

def cost(df, emb_user, emb_item):
    df["prediction"] = np.sum(emb_user[df["userID"].values] * emb_item[df["itemID"].values], axis=1)
    error = np.mean(np.square(df.prediction - df.rating))
    return error

def gradient(df, Y, emb_user, emb_item):
    R = Y.sign().todense()
    delta = np.multiply(Y.todense(),R) - sparse_multiply(df,emb_user,emb_item).todense()
    d_emb_user = -2*np.dot(delta,emb_item)/len(Y.data)
    d_emb_item = -2*np.dot(delta.transpose(),emb_user)/len(Y.data)
    return d_emb_user,d_emb_item


def gradient_descent(df, emb_user, emb_item, iterations=100, learning_rate=0.01, df_val=None):
    Y = df2matrix(df, emb_user.shape[0], emb_item.shape[0])
    grad_u_moment, grad_m_moment = gradient(df, Y, emb_user, emb_item)
    emb_user = np.array(np.subtract(emb_user, learning_rate * grad_u_moment))
    emb_item = np.array(np.subtract(emb_item, learning_rate * grad_m_moment))
    for i in range(iterations - 1):
        grad_user, grad_item = gradient(df, Y, emb_user, emb_item)
        grad_u_moment = .9 * grad_u_moment + .1 * grad_user
        grad_m_moment = .9 * grad_m_moment + .1 * grad_item
        emb_user = np.array(
            np.subtract(emb_user, learning_rate * grad_u_moment))
        emb_item = np.array(
            np.subtract(emb_item, learning_rate * grad_m_moment))
        if i % 50 == 0:
            print("Training cost:", cost(df, emb_user, emb_item))
        if df_val is not None and i % 50 == 0:
            print("Validation cost:", cost(df_val, emb_user, emb_item))
    return emb_user, emb_item


K = 15
emb_user = create_embedings(num_users, K)
emb_item = create_embedings(num_item, K)


msk = np.random.rand(len(df)) < 0.8
train = df[msk].copy()
val = df[~msk].copy()



emb_user, emb_item = gradient_descent(train, emb_user, emb_item, iterations=500, learning_rate=1,df_val=val)

user = pd.DataFrame(emb_user)
user['idx'] = user.index
user['userID'] = user.idx.apply(lambda x: idx2user[x])


#recommendaton for item ID = 1
itemID = 1

item_emb = emb_item[itemID]
item_emb.shape


user['score'] = user.apply(lambda x: np.dot(x[: K], item_emb), axis=1)
user_rank = user.sort_values(by = 'score',ascending=False)

user_rank.score.describe()
























