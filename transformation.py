#import for transformation
import numpy as np
import math
import eulerangles as eu

#import for image generation
import cv2
import xmltodict
# read a image with corp

# transform it.

sample_6para = [[0,0,0,0,0,0], [5,0,0,0,0,0], [0, 5, 0, 0, 0, 0]]

def transform(V, Mt, Mr):
	return np.dot(Mr, V + Mt)

def transform6para(V, transx = 0, transy = 0, transz = 0, rotz = 0, roty = 0, rotx = 0):
	Mt = [transx, transy, transz]
	Mr = eu.euler2mat(rotz, roty, rotx)
	return np.dot(Mr, V +  Mt)
# write the 6 num -> Mx as a function and test it.

def test_transform():
		# Test Parameter
	x, y, z, a, b, g = 1, 0, 0, math.pi/2, 0, 0
	print(x, y, z, a, b, g)


	# Construct Original V
	V = np.array([1, 0, 0])
	# V = np.array([1, 0, 0]).reshape(3,1)

	# Test func Transform
	rotz, roty, rotx = math.pi/2, 0, 0
	Mr = eu.euler2mat(rotz, roty, rotx)
	Mt = [0, 0, 0]
	print(transform(V, Mt, Mr))

	# Test func Transform6para
	print(transform6para(V, 0, 0, 0, -math.pi/2))

def target_sample():

	img = cv2.imread("test/Darren.jpg")
	calib = cv2.imread("calibration_file/calibration.jpg")
	
	height, width, channels = img.shape
	# rows, cols = image.shape
	print(height, width, channels)

	f = open("calibration_file/calibration.xml")
	dic = xmltodict.parse(f.read())

	xmin = int(dic['annotation']['object']['bndbox']['xmin'])
	ymin = int(dic['annotation']['object']['bndbox']['ymin'])
	xmax = int(dic['annotation']['object']['bndbox']['xmax'])
	ymax = int(dic['annotation']['object']['bndbox']['ymax'])	
	print(xmin,ymin,xmax,ymax)


	x_f0 = - (xmax - xmin) / 2
	y_f0 = - (ymax - ymin) / 2
	print(x_f0, y_f0)

	x_0f0_1 = 1512 + x_f0
	y_0f0_1 = 1512 + y_f0

	x_0f0_2 = 1512 - x_f0
	y_0f0_2 = 1512 + y_f0

	x_0f0_3 = 1512 + x_f0
	y_0f0_3 = 1512 - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	print(V1[0])
	print (V1, V2, V3)


	# sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
	# 2000 ~ 
	max_a = math.pi * 2
	max_b = math.pi / 2
	max_g = math.pi / 2
	max_distance = 2600
	distance_step = 200

	for item in sample_6para:
		x, y, z, a, b, g = item[0], item[1], item[2], item[3], item[4], item[5]

		print(x, y, z, a, b, g)
		V1_ = transform6para(V1, x, y, z, a, b, g)
		V2_ = transform6para(V2, x, y, z, a, b, g)
		V3_ = transform6para(V3, x, y, z, a, b, g)
	
		x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
		y_f_1 = y_f0 * V1_[1] / V1_[2] * (2000) / 135.8
		x_0f_1 = 1512 + x_f_1
		y_0f_1 = 1512 + y_f_1
		print("x_f0 is: ", x_f0, "\nx_f_1 is: ", x_f_1)
		print("y_f0 is: ", y_f0, "\ny_f_1 is: ", y_f_1)

		x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
		y_f_2 = y_f0 * V2_[1] / V2_[2] * (2000) / 135.8
		x_0f_2 = 1512 + x_f_2
		y_0f_2 = 1512 + y_f_2
		print("x_f0 is: ", x_f0, "\nx_f_2 is: ", x_f_2)
		print("y_f0 is: ", y_f0, "\ny_f_2 is: ", y_f_2)

		x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
		y_f_3 = y_f0 * V3_[1] / V3_[2] * (2000) / 135.8
		x_0f_3 = 1512 + x_f_3
		y_0f_3 = 1512 + y_f_3
		print("x_f0 is: ", x_f0, "\nx_f_3 is: ", x_f_3)
		print("y_f0 is: ", y_f0, "\ny_f_3 is: ", y_f_3)

		if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
			y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
			break
		pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3]])
		print("pts1 is: ", pts1, "\npts2 is:", pts2)
		M = cv2.getAffineTransform(pts1,pts2)
	
		dst = cv2.warpAffine(img, M, (width, height))
		img_resized = cv2.resize(img, (448, 448))
		dst_resized = cv2.resize(dst, (448, 448))
								
		cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
		# cv2.imshow('camera', img_resized)
		# cv2.waitKey()
								
		cv2.imshow('camera', dst_resized)
		cv2.waitKey()
		
		# print(pts2)
		# print(x, y, z, a, b, g)


	# Shift
	# M = np.float32([[1, 0, 200], [0, 1, 100]]) 

	# Affine Transformation
	# Test cv2 affine ransformation here

	pts1 = np.float32([[50,50],[200,50],[50,200]])
	pts2 = np.float32([[10,100],[200,50],[100,300]])
	M = cv2.getAffineTransform(pts1,pts2)
 
	dst = cv2.warpAffine(img, M, (width, height))   

	img_resized = cv2.resize(img, (448, 448))
	dst_resized = cv2.resize(dst, (448, 448))

	cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
	cv2.imshow('camera', img_resized)
	cv2.waitKey()

	cv2.imshow('camera', dst_resized)
	cv2.waitKey()



def random_sample():

	img = cv2.imread("test/Darren.jpg")
	calib = cv2.imread("calibration_file/calibration.jpg")
	
	height, width, channels = img.shape
	# rows, cols = image.shape
	print(height, width, channels)

	f = open("calibration_file/calibration.xml")
	dic = xmltodict.parse(f.read())

	xmin = int(dic['annotation']['object']['bndbox']['xmin'])
	ymin = int(dic['annotation']['object']['bndbox']['ymin'])
	xmax = int(dic['annotation']['object']['bndbox']['xmax'])
	ymax = int(dic['annotation']['object']['bndbox']['ymax'])	
	print(xmin,ymin,xmax,ymax)


	x_f0 = - (xmax - xmin) / 2
	y_f0 = - (ymax - ymin) / 2
	print(x_f0, y_f0)

	x_0f0_1 = 1512 + x_f0
	y_0f0_1 = 1512 + y_f0

	x_0f0_2 = 1512 - x_f0
	y_0f0_2 = 1512 + y_f0

	x_0f0_3 = 1512 + x_f0
	y_0f0_3 = 1512 - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	print(V1[0])
	print (V1, V2, V3)


	# sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
	# 2000 ~ 
	max_a = math.pi * 2
	max_b = math.pi / 2
	max_g = math.pi / 2
	max_distance = 2600
	distance_step = 200

	for x in range(-max_distance, max_distance, distance_step):
		# print(x)
		for y in range(-max_distance, max_distance, distance_step):
			print (x, y)
			for z in range(-1600, 1600, 200):
				for i in range(18):
					a = -max_a + max_a / 9 * i
					for j in range(18):
						b = -max_b + max_b / 9 * j
						for k in range(18):
							g = -max_g + max_g / 9 * k
							print(x, y, z, a, b, g)
							V1_ = transform6para(V1, x, y, z, a, b, g)
							V2_ = transform6para(V2, x, y, z, a, b, g)
							V3_ = transform6para(V3, x, y, z, a, b, g)

							x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
							y_f_1 = y_f0 * V1_[1] / V1_[2] * (-2000) / 92.3
							x_0f_1 = 1512 + x_f_1
							y_0f_1 = 1512 + y_f_1

							x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
							y_f_2 = y_f0 * V2_[1] / V2_[2] * (-2000) / 92.3
							x_0f_2 = 1512 + x_f_2
							y_0f_2 = 1512 + y_f_2

							x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
							y_f_3 = y_f0 * V3_[1] / V3_[2] * (-2000) / 92.3
							x_0f_3 = 1512 + x_f_3
							y_0f_3 = 1512 + y_f_3

							if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
								y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
								break
							pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3]])
							M = cv2.getAffineTransform(pts1,pts2)

							dst = cv2.warpAffine(img, M, (width, height))
							img_resized = cv2.resize(img, (448, 448))
							dst_resized = cv2.resize(dst, (448, 448))
						
							cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
							# cv2.imshow('camera', img_resized)
							# cv2.waitKey()
						
							cv2.imshow('camera', dst_resized)
							cv2.waitKey()
						
							# print(pts2)
							# print(x, y, z, a, b, g)


	# Shift
	# M = np.float32([[1, 0, 200], [0, 1, 100]]) 

	# Affine Transformation
	# Test cv2 affine ransformation here

	pts1 = np.float32([[50,50],[200,50],[50,200]])
	pts2 = np.float32([[10,100],[200,50],[100,300]])
	M = cv2.getAffineTransform(pts1,pts2)
 
	dst = cv2.warpAffine(img, M, (width, height))   

	img_resized = cv2.resize(img, (448, 448))
	dst_resized = cv2.resize(dst, (448, 448))

	cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
	cv2.imshow('camera', img_resized)
	cv2.waitKey()

	cv2.imshow('camera', dst_resized)
	cv2.waitKey()

if __name__=='__main__':
	#test_transform()
	target_sample()
	# random_sample()

