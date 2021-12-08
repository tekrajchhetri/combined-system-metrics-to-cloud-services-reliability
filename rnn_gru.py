# -*- coding: utf-8 -*-
# @Author  : Tek Raj Chhetri
# @Email   : tekraj.chhetri@sti2.at
# @Web     : http://tekrajchhetri.com/
# @File    : rnn_gru.py
# @Software: PyCharm

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM,GRU, Dense,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle


print("reading")
df = pd.read_csv("data.csv")



inputdata = df[['cpu_utilization',
             'memory_utilization', 
             'network_overhead', 
             'io_utilization',
             'bits_outputted', 
             'bits_inputted',
             'smart_188', 
             'smart_197', 
             'smart_198', 
             'smart_9', 
             'smart_1',
             'smart_5',
             'smart_187', 
             'smart_7', 
             'smart_3', 
             'smart_4',
             'smart_194',
             'smart_199']].values
targets = df["target"].values


def prepare_train_test_data(inputdata):
    T = 10 
    D = inputdata.shape[1]
    N = len(inputdata) - T 
    trainSetLength = int(len(inputdata) * 0.6)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    inputdata = imp.fit_transform(inputdata)    
    scaler = StandardScaler()
    scaler.fit(inputdata[:trainSetLength + T - 1])
    inputdata = scaler.transform(inputdata)
    X_train = np.zeros((trainSetLength, T, D))
    Y_train = np.zeros(trainSetLength)

    for t in np.arange(trainSetLength):
        X_train[t, :, :] = inputdata[t:t+T]
        Y_train[t] = targets[t+T] 
        
    X_test = np.zeros((N - trainSetLength, T, D))
    Y_test = np.zeros(N - trainSetLength)

    for u in range(N - trainSetLength):
        t = u + trainSetLength
        X_test[u, :, :] = inputdata[t:t+T]
        Y_test[u] = targets[t+T] 
    return X_train,Y_train,X_test,Y_test, T, D, N

X_train,Y_train,X_test,Y_test,T, D, N = prepare_train_test_data(inputdata)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=2000,
  decay_rate=1,
  staircase=False)

def optimizelr():
  return tf.keras.optimizers.Adam(lr_schedule)

callback_early_stop =  tf.keras.callbacks.EarlyStopping(monitor='val_loss',
										    verbose=1,
										    restore_best_weights=True)

METRICS = [
 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
   

]
print("******************************Creating Checkpointing Dir**************************************************")
checkpoint_path = "trained/gru/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    inputs = Input(shape=(T, D))
    layer = GRU(2024,  activation="relu",dropout=0.5,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5 ,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5, kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = GRU(2024,  activation="relu",dropout=0.5,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5 ,kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1012)(layer)
    layer = GRU(2024,activation="relu",dropout=0.5, kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    layer = Dense(1,  activation='sigmoid')(layer)
    model = Model(inputs,layer)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizelr(),
        metrics=METRICS
    )
    print("*********************Training***************************")
    trained_model = model.fit(
        X_train, Y_train,
        batch_size=556,
        epochs=430,
        callbacks=[callback_checkpoint,callback_early_stop],
        validation_data=(X_test, Y_test),
    )
print("************************************************************")
print("***************** Model Summary ***************************")
print(model.summary())
print("***********************************************************")
print("Finish")
