
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#rescale the rating
df_util = np.load(".npy")

#create the scorining matrix 
rp = df_util.pivot_table(cols=['item'], rows = ['id'], values = 'score')
rp = rp.fillna(0) 
rp.head()

Q = rp.values 
Q.shape 


#identify the observed points 
W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)

#parameters 
lambda_ = 0.1
n_factors = 5
m, n = Q.shape #m is the number of user, n is the number of itmes 
n_iterations = 20

X =10 * np.random.rand(m, n_factors) 
Y = 10 * np.random.rand(n_factors, n)

def get_error(Q, X, Y, W):
    return np.sqrt(np.mean((W * (Q - np.dot(X, Y)))**2))
    #return np.sqrt(np.mean(np.abs(W * (Q - np.dot(X, Y)))))

errors = []
for ii in range(n_iterations):
    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
                        np.dot(Y, Q.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
                        np.dot(X.T, Q))
    if ii % 100 == 0:
        print('{}th iteration is completed'.format(ii))
    errors.append(get_error(Q, X, Y, W))
Q_hat = np.dot(X, Y)
print('Error: {}'.format(get_error(Q, X, Y, W)))

plt.plot(errors);
plt.ylim([0, 20000])


#get the recommendations 
def recommendations(W=W, Q=Q, Q_hat=Q_hat, items=item_list):
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(10) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 10 * W, axis=1)
    recomd = {}
    for jj, item in zip(range(m), items):
        
        
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([item[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([item[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, item[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')


weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
print('Error of rating: {}'.format(get_error(Q, X, Y, W)))


recommendations(Q_hat=weighted_Q_hat)





















