#
# RCmax:  DNN using max RevComp 1st conv filter layer
#
# code modified from Quang and Xie NAR 44 e107 (2016) without LSTM
#
# score input sequences only

import numpy as np
import h5py
import scipy.io
import sys, random, os.path
np.random.seed(int(sys.argv[15])) # for reproducibility
random.seed(int(sys.argv[15]))  #used in keras/preprocessing/sequence.py
import keras.backend as K
from keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers.merge import maximum
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import layers
from keras import regularizers
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from seya.layers.recurrent import Bidirectional
#from keras.utils.layer_utils import print_layer_shapes

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

core=sys.argv[1]

maxpool1=int(sys.argv[2])
drop_1=float(sys.argv[3])
conv_filter2=int(sys.argv[4])
conv_kernel2=int(sys.argv[5])
maxpool2=int(sys.argv[6])
drop_2=float(sys.argv[7])
conv_filter3=int(sys.argv[8])
conv_kernel3=int(sys.argv[9])
maxpool3=int(sys.argv[10])
drop_3=float(sys.argv[11])
dense1=int(sys.argv[12])
lambda1=float(sys.argv[13])
lambda2=float(sys.argv[14])

datapre=sys.argv[16]
seqfile=sys.argv[17]

print('loading data')
trainmat = h5py.File(core+'/'+seqfile+'.h5','r')

print( 'trainmat traindata:',trainmat['traindata'].shape)
print( 'trainmat trainxdata:',trainmat['trainxdata'].shape)
print( 'trainmat namedata:',trainmat['namedata'].shape)

(a,b,c)=trainmat['trainxdata'].shape
seqlen=int(c/2)
nfilt=seqlen-13+1
print('seqlen:', seqlen,' nfilt: ',nfilt)

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(0,2,1))
y_train = np.array(trainmat['traindata'])

print( 'X_train:',X_train.shape)
print( 'y_train:',y_train.shape)

print('building model')

Conv1in=2*seqlen
conv_kernel1=13
conv_filter1=200
lambdain=int((Conv1in/2-conv_kernel1+1))

def maximum_ind(x):
    y = K.reverse(x,axes=1)
    z = K.maximum(x,y)
    return z[:,:lambdain,:]

def maximum_ind_shape_out(input_shape):
    out_shape = (input_shape[0],lambdain,conv_filter1)
    return out_shape

model = Sequential()

model.add(Conv1D(activation="relu",
                        input_shape=(Conv1in,4),
                        filters=conv_filter1,
                        kernel_size=conv_kernel1,
                        padding="valid",
                        strides=1))

model.add(Lambda(maximum_ind, maximum_ind_shape_out))


model.add(MaxPooling1D(strides=maxpool1, pool_size=maxpool1))

model.add(Dropout(drop_1))

Conv2in=int(lambdain/maxpool1)

model.add(Conv1D(activation="relu",
                        input_shape=(Conv2in,conv_filter1),
                        filters=conv_filter2,
                        kernel_size=conv_kernel2,
                        padding="valid",
                        strides=1))

model.add(MaxPooling1D(strides=maxpool2, pool_size=maxpool2))

model.add(Dropout(drop_2))

Conv3in=int((Conv2in - conv_kernel2 + 1) / maxpool2)

model.add(Conv1D(activation="relu",
                        input_shape=(Conv3in,conv_filter2),
                        filters=conv_filter3,
                        kernel_size=conv_kernel3,
                        padding="valid",
                        strides=1))

model.add(MaxPooling1D(strides=maxpool3, pool_size=maxpool3))

model.add(Dropout(drop_3))

model.add(Flatten())

model.add(Dense(units=dense1, kernel_regularizer=regularizers.l2(lambda2), activity_regularizer=regularizers.l1(lambda1)))
model.add(Activation('relu'))

model.add(Dense(units=1, kernel_regularizer=regularizers.l2(lambda2), activity_regularizer=regularizers.l1(lambda1)))
model.add(Activation('sigmoid'))

print( 'compiling model')
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

print( 'running at most 120 epochs')

checkpointer = ModelCheckpoint(filepath=core+"/"+datapre+"_model.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

if not os.path.isfile(core+'/'+datapre+'_model.h5'):
    print(datapre+'_model.h5 does not exist. Exiting.')
    exit()

else:
    print (datapre+"_model.h5 already exists, not retraining, just scoring.")


model.load_weights(core+'/'+datapre+'_model.h5')


print( 'predicting on sequences')
y_train = 1*np.array(trainmat['traindata'])
x = np.transpose(trainmat['trainxdata'],axes=(0,2,1))
y = model.predict(x, verbose=1)
dt = h5py.special_dtype(vlen=str)
name=np.array(trainmat['namedata'],dtype=dt)
f=open(core+'/'+seqfile+'_scores.out','w')
for i in range(len(y)):
    f.write('{0}\t{1:8.6f}\n'.format(name[i],y[i,0]))
f.close()



