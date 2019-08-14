"""
This script runs the flask api, the api call returns a dictionary. 
The main keys of the dictionary are "emotion" and "score".

The function predict_from_img, loads 3 different models, takes a tensor image and returns the emotion based on voting.
The function get_smile_meter_score uses euclidean distances of the facial landmarks to generate a smile score based on certain thresholds.

"""

import tensorflow as tf
import time
import numpy as np
import dlib
import cv2
import os

from imutils import face_utils
import scipy
from PIL import Image
from os.path import isfile, join

from models import get_model_smaller2 as get_model1
from models import get_model_smaller3 as get_model2
from models import get_model_smaller4 as get_model3

from calculate_score import get_score

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


save_model_path1 = 'Model_smaller2/'
save_model_path2 = 'Model_smaller3/'
save_model_path3 = 'Model_smaller4/'

mypath_source = "test_source_dir/images/"
mypath_end = "api_end_dir/"
mypath_smile_meter_faces = "smile_meter_faces/"


if not os.path.exists(mypath_source):
	os.makedirs(mypath_source)
if not os.path.exists(mypath_end):
	os.makedirs(mypath_end)
if not os.path.exists(mypath_end):
	os.makedirs(mypath_smile_meter_faces)


## Takes an image array, generates model predictions from 3 different models, and returns emotion class based on voting. 
def predict_from_img(image):

	tensor_image = image.reshape([-1, 48, 48, 1])
	predicted_label1 = model1.predict(tensor_image)
	predicted_label2 = model2.predict(tensor_image)
	predicted_label3 = model3.predict(tensor_image)

	emotions = ["Non-pos", "Pos"]
	label1 = list(predicted_label1[0])
	label2 = list(predicted_label2[0])
	label3 = list(predicted_label3[0])

	model1_emotion = emotions[label1.index(max(label1))]
	model1_conficence = max(label1)*100

	model2_emotion = emotions[label2.index(max(label2))]
	model2_conficence = max(label2)*100
	
	model3_emotion = emotions[label3.index(max(label3))]
	model3_conficence =  max(label3)*100
	
	
	print("Model 1 emotion---:", model1_emotion)
	print("Model 1 confidence---: ", model1_conficence)
	if model1_emotion == 'Pos' and model1_conficence < 85.0:
		model1_emotion = 'Non-pos'
		print("Model 1 emotion overturned to Non-pos......")

	print("Model 2 emotion---:", model2_emotion)
	print("Model 2 confidence---: ", model2_conficence)
	if model2_emotion == 'Pos' and model2_conficence < 85.0:
		model2_emotion = 'Non-pos'
		print("Model 2 emotion overturned to Non-pos......")

	print("Model 3 emotion---:", model3_emotion)
	print("Model 3 confidence---: ",model3_conficence)	
	if model3_emotion == 'Pos' and model3_conficence < 85.0:
		model3_emotion = 'Non-pos'
		print("Model 3 emotion overturned to Non-pos......")

	list_emotions = [model1_emotion, model2_emotion, model3_emotion]
	final_voted_emotion = max(list_emotions,key=list_emotions.count)

	return final_voted_emotion



model1 = get_model1()
model1.load_weights(save_model_path1+'best_model.h5')
model1._make_predict_function()

model2 = get_model2()
model2.load_weights(save_model_path2+'best_model.h5')
model2._make_predict_function()

model3 = get_model3()
model3.load_weights(save_model_path3+'best_model.h5')
model3._make_predict_function()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print(model1.summary())
print(model2.summary())
print(model3.summary())


def detect_faces(image):
	face_detector = dlib.get_frontal_face_detector()
	detected_faces = face_detector(image, 1)
	face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
	return face_frames


## Takes an image array, detects faces and landmarks, calculates euclidean distances of relevant facial landmarks, and generates a score based on certain thresholds.
def get_smile_meter_score(image):
	rects = detector(image, 1)
	shape = predictor(image, rects[0])
	landmark = face_utils.shape_to_np(shape)
	h1 = np.linalg.norm(landmark[60]-landmark[64])
	v1 = np.linalg.norm(landmark[61]-landmark[67])
	v2 = np.linalg.norm(landmark[62]-landmark[66])
	v3 = np.linalg.norm(landmark[63]-landmark[65])
	v = np.mean(sorted([v1, v2, v3])[-2:])
	score = get_score(h1, v)	
	return score

## Takes image path, does required preprocessing, calls functions get_smile_meter_score and predict_from_img to get emotion and score.
def predict_emotion(img_path, random_name, start):

	image = cv2.imread(img_path, 0)
	print("image shape:", image.shape)
	detected_faces = detect_faces(image)
	print(detected_faces)
	rotation = 0

	if len(detected_faces) > 0:
		for n, face_rect in enumerate(detected_faces):
			face = Image.fromarray(image).crop(face_rect)
			scipy.misc.imsave(mypath_end + random_name , face)

	try:

		name = mypath_end + random_name
		image = cv2.imread(name, 0)
		smile_meter_image = cv2.resize(image, (250, 250))
		time_till_all_preprocessing = time.time() - start

		before_smile_meter = time.time()
		score = get_smile_meter_score(smile_meter_image) 
		smile_meter_response_time = time.time() -  before_smile_meter

		image = cv2.resize(image, (48, 48))
		before_model_call_time = time.time()
		emotion = predict_from_img(image)
		model_response_time = time.time() - before_model_call_time

		return emotion, score, model_response_time, smile_meter_response_time, time_till_all_preprocessing
	except Exception as e:
		print("no face", 0, e)
		return "no face[found no image]", 0, 0, 0, time.time() - start




################################################################################################################
# load Flask 
import flask
from flask_cors import CORS
import random

app = flask.Flask(__name__)
CORS(app)



# define a predict function as an endpoint 
@app.route("/predict", methods=["POST"])
def predict():
	start = time.time()
	data = {}
	params = flask.request.json
	if (params == None):
		params = flask.request.args
		print("in if...")
	if (params != None):
		print("in else...")
		f = flask.request.files['file_data']
		random_name = str(random.randint(1,50000000)) + ".jpg"

		final_save_path = join(mypath_source, random_name)
		f.save(final_save_path)
		image_save_time = time.time()

		data["emotion"], data["score"], data["model_response_time"], data['smile_meter_response_time'], data['time_till_all_preprocessing'] = predict_emotion(final_save_path, random_name, start)
		data["file_name"] = random_name
		data["total time"] = time.time() - start

	print(data)
	return flask.jsonify(data)
app.run(host='0.0.0.0')
################################################################################################################




