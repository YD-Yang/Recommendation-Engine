
import pandas as pd
import numpy as np
import os
from surprise import Reader, dataset
from surprise import SVD, GridSearch
from surprise import evaluate, print_perf
from surprise import NMF
from surprise import SVDpp
from surprise import dump

from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle 
#######################################################################
#laod data into right format 

#reader = Reader(line_format='user item rating', rating_scale=(0.0,1.0))

# Also, a dummy Dataset class
class MyDataset(dataset.DatasetAutoFolds):

    def __init__(self, df, reader):

        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df['userID'], df['itemID'], df['rating'])]
        self.reader=reader


#rescale the rating
df = np.load(".npy")
df =pd.DataFrame(df_util, columns= ['userID','itemID','rating'])

reader = Reader(line_format='user item rating', rating_scale=(0.0, 10.0))
data = MyDataset(df, reader)

#grid search to tune parameters 
from surprise.model_selection import GridSearchCV

param_grid = { 'n_factors':[3, 5, 8, 10, 11], 'n_epochs': [1, 3, 5], 'lr_all': [0.002, .001, 0.01, .005, .0001], 
               'reg_all': [0.2, .02, .002, .001]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(data)

pd.DataFrame.from_dict(gs.cv_results)

# best RMSE score
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

print(gs.best_score['mae'])
print(gs.best_params['rmse'])


#baseline
#np.sqrt(np.mean( np.square(df['rating'] -df['rating'].mean() )))

algo =  SVD(n_factors = 5, n_epochs = 5, lr_all = 0.005, reg_all = 0.002)
data.split(n_folds=5)
evaluate(algo, data)


###############################################################################
#prediction

from collections import defaultdict


def get_top_n(predictions, n=10, threshold = 1.5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        user_cut = [i for i in user_ratings[:n] if i[1] > threshold]
        top_n[uid] = user_cut

    return top_n



from surprise.model_selection.validation import cross_validate
score = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
scoredf = pd.DataFrame(score)
mean_score = scoredf.mean()
 


trainset = data.build_full_trainset()
algo = SVD(n_factors = 5, n_epochs = 5, lr_all = 0.005, reg_all = 0.002)
algo.fit(trainset)
algo.pu


# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n_svd = get_top_n(predictions, n=3, threshold = 2.5)

# Print the recommended items for each user
for uid, user_ratings in top_n_svd.items():
    print(uid, [iid for (iid, _) in user_ratings])

#
pred_svd = open('pred_svd.pkl', 'wb')
pickle.dump(top_n_svd, pred_svd)
pred_svd.close()

# read python dict back from the file
pkl_file = open('pred_svd.pkl', 'rb')
top_n_svd = pickle.load(pkl_file)
pkl_file.close()

###############################################################################

"""
NMF
"""

df['rating'] = df['rating'].apply(lambda x: x *0.9 + 1)
df['rating'].describe()

#baseline
np.sqrt(np.mean( np.square(df['rating'] -df['rating'].mean() )))

reader = Reader(line_format='user item rating', rating_scale=(0, 10.0))
data = MyDataset(df2, reader)


param_grid = { 'n_factors':[3, 5, 8, 10, 11], 
               'n_epochs': [1, 3, 5],
               'reg_pu': [0.002, .001, 0.01, .005, .0001], 
               'reg_qi': [0.2, .02, .002, .001],  
               'reg_bu': [0.2, .02, .002, .001],
               'reg_bi': [0.2, .02, .002, .001], 
               'lr_bu': [0.2, .02, .002, .001], 
               'lr_bi': [0.2, .02, .002, .001] }

param_grid = { 'n_factors':[5], 'n_epochs': [3],'reg_qi': [.002], 'reg_bu': [.002], 
              'reg_bi': [ .001], 'lr_bu': [0.02], 'lr_bi': [0.02]}

gs = GridSearchCV(NMF, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
pd.DataFrame.from_dict(gs.cv_results)
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])



trainset = data.build_full_trainset()
algo  =  NMF()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n_svd = get_top_n(predictions, n=5)

# Print the recommended items for each user
for uid, user_ratings in top_n_svd.items():
    print(uid, [iid for (iid, _) in user_ratings])


#algo.predict(id, item, r_ui=1, verbose=True)

data.split(n_folds=5)
evaluate(algo, data, measures=['RMSE'])



""
from surprise.model_selection.validation import cross_validate
score = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
scoredf = pd.DataFrame(score)
mean_score = scoredf.mean()


"""
SVD++
"""

trainset = data.build_full_trainset()
algo  =  SVDpp()
algo.fit(trainset)

data.split(n_folds=5)
evaluate(algo, data, measures=['RMSE'])


 

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n_svd = get_top_n(predictions, n=5, threshold = 2.5)

# Print the recommended items for each user
for uid, user_ratings in top_n_svd.items():
    print(uid, [iid for (iid, _) in user_ratings])

top_n_svd['id0']
