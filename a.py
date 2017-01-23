import numpy
import random

import cPickle, zipfile, numpy, imp, theano,random,os,sys,timeit
from theano import shared
import theano.tensor as T
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adamax
from keras.regularizers import l2, activity_l2,l1
from keras.layers.normalization import BatchNormalization

# Load the dataset
####################run only once##########################3
#open a zip file, give path of the .zip file from current directory 
#f = zipfile.ZipFile('DogCatdata.zip', 'r')
#extract the zip file 
#f1 = f.extractall('DogCatdata/')
#execute the python file to get data from .mat file, give path from current directory to extracted file
#execfile('DogCatdata/DogCatdata/import_data_from_mat_file.py')   
#f.close()
##############################################################

rng = numpy.random.RandomState(30430)
#load python file data here...path of the file after extraction
foo = imp.load_source('module.name', 'DogCatdata/DogCatdata/import_data_from_mat_file.py')
#print foo.data


#batches of 500 , minimise data transfer from cpu to gpu memory
train_x1 = foo.train_data
#train_x1= train_x1.transpose(0,2,3,1)
train_y1 = foo.train_target
test_x1 = foo.test_data

'''
#to convert .mat to images back
from PIL import Image
import scipy.misc
for i  in range(train_x1.shape[0]):
    img = Image.fromarray(train_x1[i], 'RGB')
    img.save('my.png')
    scipy.misc.imsave(str(i)+'outfile.jpg', train_x1[i])'''

#convert train_y to integer
train_y1=train_y1.astype('int32')

# convert class vectors to binary class matrices                    
train_y1 = np_utils.to_categorical(train_y1,2)
train_y1=train_y1.astype('float32')


######################creating model###################################
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))
#loading stored weights
model.load_weights('smodel0.h5')


############################################################################33
#fitting into the model
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])
model.summary() 

model.fit(train_x1,train_y1, validation_split=0.50, batch_size=500, nb_epoch=6)
cur_y = model.predict(test_x1)
######################################################################################
#saving to text file
'''text_file = open("Trial.txt", "w")
for i in range(cur_y.shape[0]):
    text_file.write(str(cur_y[i]))
    text_file.write("\n")
text_file.close()'''

arr=[0]*cur_y.shape[0]

#converting to 0 and 1 again
for i in range(cur_y.shape[0]):
    if cur_y[i][0]>cur_y[i][1]:
        arr[i]=1
    else:
        arr[i]=0


#saving to text file
text_file = open("output2.txt", "w")
for i in range(cur_y.shape[0]):
    text_file.write(str(arr[i]))
    text_file.write("\n")
text_file.close()

'''cnt = 0
#counting no of 0 and 1 in test
for i in range(cur_y.shape[0]):
    if cur_y[i] == 1:
        cnt = cnt + 1
print cur_y,cnt'''

#saving the parameters
model.save('smodel.h5')
