from resize import trainX,trainY
import cPickle, zipfile, numpy, imp, theano,random,os,sys,timeit
from theano import shared
from resize import trainX,trainY,testX,testY
import theano.tensor as T
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,Adamax
from keras.regularizers import l2, activity_l2,l1
from keras.layers.normalization import BatchNormalization
import PIL
from PIL import Image
from scipy.misc import imread
from os import listdir
from os.path import isfile, join


objects = ['human','object']

##############################################################

#rng = numpy.random.RandomState(30430)
"""
#load python file data here...path of the file after extraction
foo = imp.load_source('module.name', 'cifar-100-matlab/import_data_from_mat_file.py')
#print foo.data


#batches of 500 , minimise data transfer from cpu to gpu memory
train_x1 = foo.train_data
#train_x1= train_x1.transpose(0,2,3,1)
train_y1 = foo.train_target
test_x1 = foo.test_data

'''
from PIL import Image
import scipy.misc
for i  in range(train_x1.shape[0]):
    img = Image.fromarray(train_x1[i], 'RGB')
    img.save('my.png')
    scipy.misc.imsave(str(i)+'outfile.jpg', train_x1[i])'''
"""
#convert train_y to integer
trainY=trainY.astype('int32')

# convert class vectors to binary class matrices                    
trainY = np_utils.to_categorical(trainY)
trainY=trainY.astype('float32')

#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
print trainY.shape[1]

######################creating model###################################
model = Sequential()
model.add(Convolution2D(12, 3, 3, input_shape=(3, 70, 70)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(12, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(20, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(15))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))
#model.load_weights('smodel.h5')


############################################################################33
#fitting into the model
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])
model.summary() 

model.fit(trainX,trainY, validation_split=0.2, batch_size=100, nb_epoch=10)
model.save('smodel.h5')
prediction = model.predict(testX)
"""
#saving to text file
"""
cnt = 0
cnt1 = 0
pred = []
text_file = open("Trial.txt", "w")
text_file1 = open("Trial1.txt", "w")
for i in range(prediction.shape[0]):
    text_file.write(str(prediction[i]))
    text_file.write("\n")
    if prediction[i][0]>prediction[i][1]:
    	pred.append(0)
    	text_file1.write('0')
        text_file1.write("\t")
        cnt = cnt +1
    else:
    	pred.append(1)
    	text_file1.write('1')
        text_file1.write("\t")
        cnt1 = cnt1+1


print cnt,cnt1
s = int(sum(testY))
error = float(s-cnt1)/float(s)
print s,error
text_file.close()
text_file1.close()


#index = numpy.argmax(prediction)


#saving the parameters
#model.save('smodel.h5')'''
