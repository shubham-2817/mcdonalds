#from parameters import DATASET, NETWORK
import numpy as np


class Dataset:
	name = 'Fer2013'
	train_folder = '/training'


class Network:
	input_size = 48
	output_size = 2
	activation = 'relu'
	loss = 'categorical_crossentropy'
	use_batchnorm_after_conv_layers = True
	use_batchnorm_after_fully_connected_layers = False
	use_landmarks = True
	use_hog_and_landmarks = True
	use_hog_sliding_window_and_landmarks = True


DATASET = Dataset()
NETWORK = Network()

DATASET.trunc_trainset_to = 0

def load_data(validation=False, test=False):
	
	data_dict = dict()
	validation_dict = dict()
	test_dict = dict()

	if DATASET.name == "Fer2013":

		data_dict['X'] = np.load(DATASET.train_folder + '/images.npy')
		data_dict['X'] = data_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
		if NETWORK.use_landmarks:
			data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
		if not  NETWORK.use_hog_and_landmarks:
			data_dict['X2'] = np.load(DATASET.train_folder + '/landmarks.npy')
			data_dict['X2'] = np.array([x.flatten() for x in data_dict['X2']])
			data_dict['X2'] = np.concatenate((data_dict['X2'], np.load(DATASET.train_folder + '/hog_features.npy')), axis=1)
		data_dict['Y'] = np.load(DATASET.train_folder + '/labels.npy')
		if DATASET.trunc_trainset_to > 0:
			data_dict['X'] = data_dict['X'][0:DATASET.trunc_trainset_to, :, :]
			if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
				data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :]
			elif NETWORK.use_landmarks:
				data_dict['X2'] = data_dict['X2'][0:DATASET.trunc_trainset_to, :, :]
			data_dict['Y'] = data_dict['Y'][0:DATASET.trunc_trainset_to, :]

		if validation:
			# load validation set
			validation_dict['X'] = np.load(DATASET.validation_folder + '/images.npy')
			validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
			if NETWORK.use_landmarks:
				validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
			if NETWORK.use_hog_and_landmarks:
				validation_dict['X2'] = np.load(DATASET.validation_folder + '/landmarks.npy')
				validation_dict['X2'] = np.array([x.flatten() for x in validation_dict['X2']])
				validation_dict['X2'] = np.concatenate((validation_dict['X2'], np.load(DATASET.validation_folder + '/hog_features.npy')), axis=1)
			validation_dict['Y'] = np.load(DATASET.validation_folder + '/labels.npy')
			if DATASET.trunc_validationset_to > 0:
				validation_dict['X'] = validation_dict['X'][0:DATASET.trunc_validationset_to, :, :]
				if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
					validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :]
				elif NETWORK.use_landmarks:
					validation_dict['X2'] = validation_dict['X2'][0:DATASET.trunc_validationset_to, :, :]
				validation_dict['Y'] = validation_dict['Y'][0:DATASET.trunc_validationset_to, :]
		
		if test:
			# load test set
			test_dict['X'] = np.load(DATASET.test_folder + '/images.npy')
			test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
			if NETWORK.use_landmarks:
				test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
			if NETWORK.use_hog_and_landmarks:
				test_dict['X2'] = np.load(DATASET.test_folder + '/landmarks.npy')
				test_dict['X2'] = np.array([x.flatten() for x in test_dict['X2']])
				test_dict['X2'] = np.concatenate((test_dict['X2'], np.load(DATASET.test_folder + '/hog_features.npy')), axis=1)
			test_dict['Y'] = np.load(DATASET.test_folder + '/labels.npy')
			if DATASET.trunc_testset_to > 0:
				test_dict['X'] = test_dict['X'][0:DATASET.trunc_testset_to, :, :]
				if NETWORK.use_landmarks and NETWORK.use_hog_and_landmarks:
					test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :]
				elif NETWORK.use_landmarks:
					test_dict['X2'] = test_dict['X2'][0:DATASET.trunc_testset_to, :, :]
				test_dict['Y'] = test_dict['Y'][0:DATASET.trunc_testset_to, :]

		# if not validation and not test:
		#	 return data_dict
		# elif not test:
		#	 return data_dict, validation_dict
		# else: 
		#	 return data_dict, validation_dict, test_dict



		# data_X_new = np.concatenate((data_dict['X'],validation_dict['X'],test_dict['X']), axis= 0)
		# data_X2_new = np.concatenate((data_dict['X2'],validation_dict['X2'],test_dict['X2']), axis= 0)
		# data_Y_new = np.concatenate((data_dict['Y'],validation_dict['Y'],test_dict['Y']), axis= 0) 

	data_X_new =  data_dict['X']
	data_X_new = data_X_new/255.0

	data_X2_new = data_dict['X2']
	data_X2_new = data_X2_new.reshape((data_X_new.shape[0], -1))

	data_Y_new = data_dict['Y']

	print(data_Y_new[:10])
	data_Y_new = [np.where(r==1)[0][0] for r in data_Y_new]
	print(data_Y_new[:10])
	data_Y_new = [0 if x==2 else x for x in data_Y_new]
	print(data_Y_new[:10])

	from keras.utils import to_categorical
	data_Y_new = to_categorical(data_Y_new)
	print(data_Y_new[:10])

	val_test_size = 8000
	test_index = 5000
	'''
		random_indices = np.random.choice(data_X_new.shape[0], val_test_size)
		remaining_indices = np.array(list(set(list(np.arange(data_X_new.shape[0]))) - set(list(random_indices))))
	'''
	random_indices2 = np.random.choice(data_X_new.shape[0], data_X_new.shape[0], replace=False)
	remaining_indices = random_indices2[:-8000]

	random_indices = random_indices2[-8000:]
	print(len(random_indices2))
	print(len(remaining_indices))

	random_indices = random_indices2[-8000:]
	print(len(random_indices))
	print(data_X_new.shape,data_X2_new.shape,data_Y_new.shape, len(random_indices), len(remaining_indices), len(random_indices2))

	data_dict['X'] = data_X_new[remaining_indices]
	validation_dict['X'] = data_X_new[random_indices[:test_index]]
	test_dict['X'] = data_X_new[random_indices[test_index:]]
	
	print(len(data_X_new), len(data_dict['X']), len(validation_dict['X']), len(test_dict['X']))		
	data_dict['X2'] = data_X2_new[remaining_indices]
	validation_dict['X2'] = data_X2_new[random_indices[:test_index]]
	test_dict['X2'] = data_X2_new[random_indices[test_index:]]
	
	data_dict['Y'] = data_Y_new[remaining_indices]
	validation_dict['Y'] = data_Y_new[random_indices[:test_index]]
	test_dict['Y'] = data_Y_new[random_indices[test_index:]]		
	
	print(data_dict['X'].shape, validation_dict['X'].shape, test_dict['X'].shape)
		

	return data_dict, validation_dict, test_dict

	# else:
	# 	print( "Unknown dataset")
	# 	exit()

load_data(validation=False, test=False)
