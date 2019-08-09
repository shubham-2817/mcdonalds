# from os import *
# from os.path import isfile, join
# image_files = [f for f in listdir(processed_test_images_path) if isfile(join(processed_test_images_path, f))]

import random
def get_score(h1, v):
	if h1 <= 77:
		if v <= 5:
			score = random.randint(10, 30)
		elif( 5 < v <= 10):
			score = random.randint(20, 40)
		elif( 10 < v <= 20):
			score = random.randint(30, 50)
		elif( v > 40):
			score = random.randint(50, 70)
		else:
			score = random.randint(50, 70) 
	elif 77 < h1 <= 90:
		if v <= 5:
			score = random.randint(10, 30)
		elif( 5 < v <= 8):
			score = random.randint(30, 50)
		elif( 8 < v <= 10):
			score = random.randint(40, 60)
		elif( 10 < v <= 20):
			score = random.randint(60, 90)
		elif( 20 < v <= 25):
			score = random.randint(90, 100)
		elif( v > 40):
			score = random.randint(50, 70)
		else:
			score = random.randint(100, 125)
	elif 90 < h1 <= 95:
		if v <= 0.49:
			score = random.randint(10, 30)
		elif( 0.5 < v <= 2):
			score = random.randint(30, 50)
		elif( 2 < v <= 5):
			score = random.randint(40, 60)
		elif( 5 < v <= 10):
			score = random.randint(50, 70)			
		elif( 10 < v <= 20):
			score = random.randint(60, 90)
		elif( 20 < v <= 25):
			score = random.randint(90, 100)
		else:
			score = random.randint(100, 125)
	elif 95 < h1 <= 100:
		if v <= 0.49:
			score = random.randint(10, 30)
		elif( 0.5 < v <= 2):
			score = random.randint(30, 50)
		elif( 2 < v <= 5):
			score = random.randint(40, 60)
		elif( 5 < v <= 10):
			score = random.randint(60, 70)			
		elif( 10 < v <= 20):
			score = random.randint(70, 90)
		elif( 20 < v <= 25):
			score = random.randint(90, 100)
		else:
			score = random.randint(100, 125)						
	elif 100 < h1 <= 105:
		if v <= 3:
			score = random.randint(30, 50)
		elif 3 < v <= 5:
			score = random.randint(50, 70)
		elif( 5 < v <= 7):
			score = random.randint(70, 90)
		elif( 7 < v <= 15):
			score = random.randint(90, 110)
		elif( 15 < v <= 20):
			score = random.randint(100, 115)
		else:
			score = random.randint(120, 150)

	elif 105 < h1 <= 115:
		if v <= 3:
			score = random.randint(30, 70)
		elif 3 < v <= 5:
			score = random.randint(70, 100)
		elif( 5 < v <= 10):
			score = random.randint(90, 110)
		elif( 10 < v <= 15):
			score = random.randint(100, 120)			
		elif( 15 < v <= 20):
			score = random.randint(110, 125)
		else:
			score = random.randint(120, 140)
	else:
		if v <= 5:
			score = random.randint(70, 90)
		elif 5 < v <= 10:
			score = random.randint(90, 120)
		else:
			score = random.randint(120, 140)		
	# print("score",score)
	return score