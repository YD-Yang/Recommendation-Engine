

####################################################################################
#backend propration 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import urllib
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 700, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")



def is_num(x):
    x = df[x]
    try:
        if x.dtype == np.int64 or x.dtype == np.float64:
            return True
        else:
            return False
    except AttributeError:
        return False
    
def is_complete(x):
    if x == 'score': return False
    return pd.isnull(df[x]).sum()== 0

categorical = list(filter(lambda x: not is_num(x) and is_complete(x), list(df)))
continuous = list(filter(lambda x: is_num(x) and is_complete(x), list(df)))


#from sklearn import datasets, metrics, preprocessing
cat_layers = []
real_layers = []
deep = []
wide = []

for x in categorical:
    cat_layers.append(tf.contrib.layers.sparse_column_with_keys(x, keys=set(df[x]), combiner='sqrtn'))    
for x in continuous:
    real_layers.append(tf.contrib.layers.real_valued_column(x, dimension=1, dtype=tf.float32))
for x in cat_layers:
    deep.append(tf.contrib.layers.embedding_column(x,dimension=8))
    wide.append(x)
for x in real_layers:
    deep.append(x)



def input_fn(df, train=True):
    

    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                        for k in continuous}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in categorical}
   
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)

    label = None
    if train:
        label = tf.constant(df[label].values)

    return feature_cols, label

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test, train=False)


####################################################################################
# fit linear model 
    
label = 'score_adj'
model_var = categorical + continuous + [label] +  ['id' ]
aa = df[continuous].corr()

    
#model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
model_dir = "\model_v5"
print("model directory = %s" % model_dir)
    

estimator = tf.contrib.learn.DNNLinearCombinedRegressor(
    model_dir = model_dir,
    # wide settings
    linear_feature_columns=wide,
    linear_optimizer=tf.train.FtrlOptimizer(
                                        learning_rate=0.1,
                                        l1_regularization_strength=0.001,
                                        l2_regularization_strength=0.001),
    # deep settings
    dnn_feature_columns=deep,
    dnn_hidden_units=[64, 32, 8],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                        learning_rate=0.05,
                                        l1_regularization_strength=0.001,
                                        l2_regularization_strength=0.001)
)

#remember to chance the label variable 
def input_fn(df, train=True):
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                        for k in continuous}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in categorical}
   
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)

    label = None
    if train:
        label = tf.constant(df['score_adj'].values)

    return feature_cols, label


df_train, df_test = train_test_split(df[model_var], random_state = 0)
#df_train.fillna('', inplace=True)
#df_test.fillna('', inplace=True)



estimator.fit(input_fn=train_input_fn, steps=1000)

#check the model fit 
evals = estimator.evaluate(input_fn=lambda: input_fn(df_train), steps=1)

#create predicted label 
pred = estimator.predict(input_fn=train_input_fn)
pred = list(pred)

#change to to original scale 
aa = [lgt_inv(x) for x in pred]
bb = [ lgt_inv(x) for x in df_train['score_adj']]

#rmse 
np.sqrt(np.mean(np.power(np.array(bb) - np.array(aa), 2)))


for i in range(len(aa)):
    if aa[i] != aa[i]:
        print(i)


pred_adj = np.array(pred)
def trunc(x):
    if x <= 0 :
        val = 0
    elif x >=1:
        val = 1 
    else:
        val = x
    return val     
pred_adj =[ trunc(x) for x in pred]

#rmse 
np.sqrt(np.mean(np.power(np.array(df_test['score']) - np.array( pred_adj), 2)))


####################################################################################
#fit categorical model 
threshold = 0.15
df['label'] =  df.score.map(lambda x: 0 if x <= threshold else 1)

label = 'label'
model_var = categorical + continuous + [label] +  ['id' ]
aa = df[continuous].corr()
    
#model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
model_dir = "model_v4"
print("model directory = %s" % model_dir)
    

estimator = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = model_dir,
    # wide settings
    linear_feature_columns=wide,
    linear_optimizer=tf.train.FtrlOptimizer(
                                        learning_rate=0.1,
                                        l1_regularization_strength=0.001,
                                        l2_regularization_strength=0.001),
    # deep settings
    dnn_feature_columns=deep,
    dnn_hidden_units=[64, 32],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                        learning_rate=0.05,
                                        l1_regularization_strength=0.001,
                                        l2_regularization_strength=0.001)
)



#remember to chance the label variable 
def input_fn(df, train=True):
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                        for k in continuous}

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
                        for k in categorical}
   
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)

    label = None
    if train:
        label = tf.constant(df['label'].values)

    return feature_cols, label


df_train, df_test = train_test_split(df[model_var], random_state = 0)


estimator.fit(input_fn=train_input_fn, steps=500)

#check the model fit 
evals = estimator.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
#evals2 = model.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

#create predicted label 
pred = estimator.predict(input_fn=eval_input_fn)
pred = list(pred)
    
#create predicted probabilities 
pred_prob = estimator.predict_proba(input_fn=eval_input_fn)
probs = []
for item in list(pred_prob):
    probs.append(list(item)[1])

    
out = list(zip(list(df_test['sdr_person_id']), pred_label, probs ))
cols = ['sdr_person_id', 'pred_label', 'pred_prob']
df_out = pd.DataFrame(out, columns=cols)
df_out = df_out.merge(df_test[['sdr_person_id',  'label']], on = 'sdr_person_id' )


