#########################################################################
#Script Name : main.py
#Description : 1) Data preparation
#              2) Bulding Deep Hash Model
#              3) Model Evaluation
#Python Ver  : 3.6
#########################################################################
import copy
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, merge, Input, BatchNormalization,\
	LocallyConnected2D, Lambda
from keras.optimizers import  Adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras import backend as K
import theano.tensor as T
import numpy as np
import os, pickle
import time
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.layers import LeakyReLU
from keras.layers.advanced_activations import PReLU
import * from Evaluation

#please input raw data before run
directory = \\\directory of raw data\\\
pickle_in = open(os.path.join(directory ,"name.pickle"),"rb")
X = pickle.load(pickle_in)

pickle_in = open(os.path.join(directory ,"name.pickle"),"rb")
y = pickle.load(pickle_in)

Y = np_utils.to_categorical(y,124)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1
	x = np.clip(x, -1, 1)
	return x

def loss_01_ouput_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2
	shape[-1] = 1
	return tuple(shape)


def hash_01_loss(y_true, y_pred):
	return y_pred

    
def DRGH_Net(hash_num):
    main_input = Input(shape=(1,128,88), name='main_input')
    act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
    C1 = Convolution2D(16, (3, 3), padding='same',dim_ordering="th")(main_input)
    B1 = BatchNormalization(axis=1)(C1)
    A1 =LeakyReLU(0.2)(C1)
    M1 = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering="th")(A1)
    
    
    C2 = Convolution2D(32, (3, 3),padding='same')(M1)
    A2 = LeakyReLU(0.2)(C2)
    M2 = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering="th")(A2)
    
    C3 = Convolution2D(64,( 3, 3),padding='same', dim_ordering="th")(M2)
    A3 = LeakyReLU(0.2)(C3)
    M3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(A3)

    C4 = Convolution2D(124, (3, 3),padding='same',dim_ordering="th")(M3)
    A4 = LeakyReLU(0.2)(C4)
    C4_flatten = Flatten()(A4)
    
    
    Deepid_layer = Dense(1024, name='feature_layer')(C4_flatten)
    A5 = LeakyReLU(0.2)(Deepid_layer)
    Hash_layer = Dense(hash_num, name='hash_layer')(A5)
    A6 = Activation('tanh',name='A6')(Hash_layer)
    Softmax_layer = Dense(100, activation='softmax', name='softmax_layer')(A6)
    loss_01_layer = Lambda(loss_01, output_shape= loss_01_ouput_shape, name='loss_01_layer')(A6)
    model = Model(input=main_input, output=[Softmax_layer, loss_01_layer])
    return model
    
#Train the model

hash_num=16
model_train(x_train,y_train):
adam = Adam()
model = DRGH_Net(hash_num)
model.summary()
model.compile(loss={'loss_01_layer':hash_01_loss, 'softmax_layer':'categorical_crossentropy'},
			   optimizer=adam,loss_weights={'loss_01_layer':0.01, 'softmax_layer':1.0}, metrics={'softmax_layer':'accuracy'})
hist = model.fit({'main_input':x_train}, {'loss_01_layer':np.zeros((y_train.shape[0],1)), 'softmax_layer':y_train},
					 batch_size=32 ,shuffle=True, nb_epoch=100, validation_split=0.1)
                     




y_train=y_train.argmax(axis=-1)
y_train=y_train.astype('int32')
y_test=y_test.argmax(axis=-1)
y_test=y_test.astype('int32')
train_set_y = copy.deepcopy(y_train)
test_set_y = copy.deepcopy(y_test)


Deephash_output = Model(input=model.get_layer('main_input').input,output=model.get_layer('A6').output)
train_set_x = Deephash_output.predict(x_train)
test_set_x = Deephash_output.predict(x_test)
gallery_binary_x = T.sgn(train_set_x).eval()
query_binary_x = T.sgn(test_set_x).eval()

train_binary_x, train_data_y = gallery_binary_x, train_set_y
train_data_y.shape = (train_set_y.shape[0], 1)
test_binary_x, test_data_y = query_binary_x, test_set_y
test_data_y.shape = (test_set_y.shape[0], 1)

train_y_rep = repmat(train_data_y, 1, test_data_y.shape[0])
test_y_rep = repmat(test_data_y.T, train_data_y.shape[0], 1)
cateTrainTest = (train_y_rep == test_y_rep)
train_data_y = train_data_y + 1
test_data_y = test_data_y + 1

train_data_y = np.asarray(train_data_y, dtype=int)
test_data_y = np.asarray(test_data_y, dtype=int)


B = compactbit(train_binary_x)
tB = compactbit(test_binary_x)

hammRadius = 2
hammTrainTest = hammingDist(tB, B).T
start_time=time.time()


Ret = (hammTrainTest <= hammRadius + 0.000001)
[Pre, Rec] = evaluate_macro(cateTrainTest, Ret)
print ('Precision with Hamming radius_2 = ', Pre)
print ('Recall with Hamming radius_2 = ', Rec)

HammingRank = np.argsort(hammTrainTest, axis=0)
[MAP, p_topN] = cat_apcal(train_data_y, test_data_y, HammingRank,10)
print ('MAP with Hamming Ranking = ', MAP)
print ('Precision of top %d returned = %f ' % (10, p_topN))
