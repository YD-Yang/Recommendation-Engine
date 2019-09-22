

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split 

#importing data


df = pd.read_csv(".csv")

 
training_set, test_set = train_test_split(df1, test_size = 0.2, random_state = 10 )
training_set = np.array(object=training_set) 
test_set = np.array(object=test_set) 

#converting training and test dataset to torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#convert ratings into 1 or 0 (classification)
#Not rated  convert from 0 to -1

threshold = 0.15
training_set[training_set == 0] = -1
training_set[(training_set <= threshold) & (training_set >= 0) ] = 0
training_set[training_set > threshold] = 1

#test set
test_set[test_set == 0] = -1
test_set[(test_set <= threshold) & (test_set >= 0)] = 0
test_set[test_set > threshold] = 1

#################################################################################################################

#architecture of RBM
# Creating the architecture of the Neural Network
#architecture of RBM

class RBM():
    def __init__(self, nv, nh): #no of vissible and hidden nodes
        #initialize weights and bias
        self.W = torch.randn(nh, nv) #random normal dist.
        #create two bias, prob of hidden node given visible node, 
        #and prob of visible node given hidden nodes
        self.a = torch.randn(1, nh) #1 is batch size since we cannot create 1d tensor in torch
        self.b = torch.randn(1, nv)
        
    #sampling Hidden node given visible - sigmoid activation function
    #i.e prob of hidden node = 1 given visible nodes
    def sampleh(self, x):#x is visible neurons
        wx = torch.mm(x, self.W.t()) 
        zx = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(zx)
        return p_h_given_v, torch.bernoulli(p_h_given_v) #bernoulli - cutoff
    
    def samplev(self, y):#x is hidden neurons
        wy = torch.mm(y, self.W)
        zy = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(zy)
        return p_v_given_h, torch.bernoulli(p_v_given_h) 
    
    def train(self, v0, vk, ph0, phk): #v0 - input vector of single user 
        self.W +=  (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0).view(1, nv)
        self.a += torch.sum((ph0 - phk), 0).view(1, nh)
        
   
nv = len(training_set[0])
nh = 10
batch_size = 100
rbm = RBM(nv, nh)

nb_users = training_set.shape[0]
nb_items = training_set.shape[1]

#training the RBM model
nb_epochs = 10
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk =  training_set[id_user : id_user + batch_size]
        v0 =  training_set[id_user : id_user + batch_size]
        ph0,_ = rbm.sampleh(v0)
        for k in range(10):
            _, hk = rbm.sampleh(vk) #we take vk since v0 and vk are same at begingin anf v0 is the target which shld not be updated
            _, vk = rbm.samplev(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sampleh(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0])) #mean square error, since abs is equal to square is not equal 
        s += 1.
    print('epoch:'+ str(epoch)+ ' loss:' + str(train_loss/s))
    


#Testing RBM model
ub_test = test_set.shape[0]
test_loss = 0.
s = 0.
for id_user in range(ub_test):
    v =  test_set[id_user : id_user + 1] #make prediction on training
    vt = test_set[id_user : id_user + 1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sampleh(v) #we take vk since v0 and vk are same at begingin anf v0 is the target which shld not be updated
        _, v = rbm.samplev(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s += 1.
print('test loss:' + str(test_loss/s))



###check the baseline loss of the algorithms 
train_loss0= 0
s = 0
for id_user in range(nb_users):
    vt = test_set[id_user : id_user + 1]
    if len(vt[vt>=0]) > 0:
        train_loss0 += torch.mean(torch.abs(vt[vt>=0]))
        s += 1.
print('train loss:' + str(train_loss0/s))
         










    
