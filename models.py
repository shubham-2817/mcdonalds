"""
This file contains different smile detection models.
Other than the function- get_model_with_lmks, all the models dont use the facial landmarks.
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.layers import *

from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, SGD
from keras.optimizers import *
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


num_classes = 2

epochs = 100
lrate = 0.01
decay = lrate/5

def get_model():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 100
#	lrate = 0.01
#	decay = lrate/5
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model



def get_model_smaller():
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 100
#	lrate = 0.01
#	decay = lrate/5
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model



def get_model_smaller2():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model



def get_model_smaller3():
	model = Sequential()
	model.add(Conv2D(16, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model



def get_model_smaller4():
	model = Sequential()
	model.add(Conv2D(16, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model




def get_model_smaller5():
	model = Sequential()
	model.add(Conv2D(16, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))	
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model




def get_model_smaller6():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))	
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	return model



def get_model_with_lmks():
	input_x = Input(shape=(48,48,1))
	x = Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same')(input_x)
	x = BatchNormalization()(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Flatten()(x)
	x = Dropout(0.4)(x)
	x = Dense(256, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = Dense(32, activation='relu')(x)

	input_l = Input(shape=(136,))
	l = Dense(64, kernel_initializer='normal', activation='relu')(input_l)
	l = Dense(32, activation='relu')(l)

	mergedOut = add([x,l])
	mergedOut = Dense(64, activation='relu')(mergedOut)
	mergedOut = Dense(num_classes, activation='softmax')(mergedOut)

	model = Model(inputs=[input_x, input_l], outputs=mergedOut)
#	epochs = 100
#	lrate = 0.01
#	decay = lrate/5
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
	return model



def get_model_without_lmks():
	input_x = Input(shape=(48, 48, 1))
	x = Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same')(input_x)
	x = BatchNormalization()(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)	
	x = Flatten()(x)
	x = Dropout(0.4)(x)
	x = Dense(256, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	x = Dense(num_classes, activation='softmax')(x)	
	mergedOut = Dense(num_classes, activation='softmax')(x)
	model = Model(inputs=input_x, outputs=mergedOut)
#	epochs = 50
#	lrate = 0.01
#	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	optimizer = Adam(lr=0.016, beta_1=0.95, beta_2=0.864, decay=decay)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
