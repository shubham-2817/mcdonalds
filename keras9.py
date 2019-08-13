import time
import argparse
import os
import numpy as np
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_loader5 import load_data 
from models import get_model_smaller4 as get_model


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


batch_size = 128
epochs = 100
save_model_path = "Model_smaller4/"
# save_model_path = "Model_smaller4_with_sgd/"

if not os.path.exists(save_model_path):
	os.makedirs(save_model_path)

def train(epochs=epochs, batch_size=batch_size,train_model=True):

	data, validation, test = load_data(validation=False, test=False)
	model = get_model()

	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
	mc = ModelCheckpoint(save_model_path + 'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

	model_info = model.fit(data['X'], data['Y'], epochs=epochs, batch_size=batch_size, validation_data=(validation['X'], validation['Y']),callbacks=[es, mc])

	y_val = model.predict(validation['X'])
	print(classification_report(np.argmax(validation['Y'], axis= 1), np.argmax(y_val, axis= 1)))



def evaluate():
	data, validation, test = load_data(validation=False, test=False)
	model = get_model()

	model.load_weights(save_model_path+'best_model.h5')

	y_test = model.predict(test['X'])
	print(classification_report(np.argmax(test['Y'], axis= 1), np.argmax(y_test, axis= 1)))



# parse arg to see if we need to launch now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")

args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
	train()
if args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES":
	evaluate()


