#import for transformation
import numpy as np
import math
import eulerangles as eu

#import for image generation
import cv2
import xmltodict
# read a image with corp

# transform it.

HALF_SZ = 224
# HALF_SZ = 1512
SCALE = 224.0 / 1512.0

sample_6para_expand = []
sample_6para = [ \
# Random Sample Scope Define.

# # Sample Translation for Original Photo
[0,0,0,0,0,0], \
[5,0,0,0,0,0], \
[30,0,0,0,0,0], \
[60, 0,0,0,0,0], \
[100, 0,0,0,0,0], \
[200, 0,0,0,0,0], \
[300, 0,0,0,0,0], \

[0, 30,0,0,0,0], \
[0, 60,0,0,0,0], \
[0, 100,0,0,0,0], \

[0, 0,300,0,0,0], \
[0, 0,600,0,0,0], \
[0, 0,900,0,0,0], \

# sample success: when sticker move some step along X, picture go right,
# when move some step along Y, picture go up
# when move some step along Z, picture shrink


# Sample Rotation for Original Photo
[0, 0,700,-math.pi/12*5,0,0], \
[0, 0,700,-math.pi/3,0,0], \
[0, 0,700,-math.pi/4,0,0], \
[0, 0,700,-math.pi/6,0,0], \
[0, 0,700,-math.pi/12,0,0], \
[0, 0,700,0,0,0], \
[0, 0,700,math.pi/12,0,0], \
[0, 0,700,math.pi/6,0,0], \
[0, 0,700,math.pi/4,0,0], \
[0, 0,700,math.pi/3,0,0], \
[0, 0,700,math.pi/12*5,0,0], \

[0, 0,700,0,-math.pi/12*5,0], \
[0, 0,700,0,-math.pi/3,0], \
[0, 0,700,0,-math.pi/4,0], \
[0, 0,700,0,-math.pi/6,0], \
[0, 0,700,0,-math.pi/12,0], \
[0, 0,700,0,0,0], \
[0, 0,700,0,math.pi/12,0], \
[0, 0,700,0,math.pi/6,0], \
[0, 0,700,0,math.pi/4,0], \
[0, 0,700,0,math.pi/3,0], \
[0, 0,700,0,math.pi/12*5,0], \
# [0, 0,900,0,math.pi/2,0], \

[0, 0,700,0,0,-math.pi/12*5], \
[0, 0,700,0,0,-math.pi/3], \
[0, 0,700,0,0,-math.pi/4], \
[0, 0,700,0,0,-math.pi/6], \
[0, 0,700,0,0,-math.pi/12], \
[0, 0,700,0,0,0], \
[0, 0,700,0,0,math.pi/12], \
[0, 0,700,0,0,math.pi/6], \
[0, 0,700,0,0,math.pi/4], \
[0, 0,700,0,0,math.pi/3], \
[0, 0,700,0,0,math.pi/12*5], \
# [0, 0,900,0,0,math.pi/2], \

# sample success: when sticker rotate positively alpha around z-axis, the sticker on the image turns around z-axis unclockwise.
# when sticker rotate positively beta aound y-axis, the sticker on the image turns around y-axis negatively(mirror)
# when sticker rotate positively gamma around x-axis, the sticker on the image turns around x-axis negatively(mirror)

# analysis: actually for the rotation of a plane, it doesn't matter it is rotated clockwisely or unclockwisely.
# [0, 5, 0, 0, 0, 0] \
]



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

	sample_matrixes = []
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


	x_f0 = - (xmax - xmin) / 2 * SCALE
	y_f0 = - (ymax - ymin) / 2 * SCALE
	print(x_f0, y_f0)

	x_0f0_1 = HALF_SZ + x_f0
	y_0f0_1 = HALF_SZ + y_f0

	x_0f0_2 = HALF_SZ - x_f0
	y_0f0_2 = HALF_SZ + y_f0

	x_0f0_3 = HALF_SZ + x_f0
	y_0f0_3 = HALF_SZ - y_f0

	x_0f0_4 = HALF_SZ - x_f0
	y_0f0_4 = HALF_SZ - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3], [x_0f0_4,y_0f0_4]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0 * SCALE
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	x4, y4, z4 = 92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	V4 = np.array([x4, y4, z4])
	print(V1[0])
	print (V1, V2, V3, V4)


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

		# print ("V1, V2, V3 is:", V1, V2, V3)

		'''rotate in self coordinate system'''
		V1_self = np.array([V1[0], V1[1], 0])
		V2_self = np.array([V2[0], V2[1], 0])
		V3_self = np.array([V3[0], V3[1], 0])
		V4_self = np.array([V4[0], V4[1], 0])
		# print ("V1, V2, V3 self is:", V1_self, V2_self, V3_self)

		V1_self_ = transform6para(V1_self, 0, 0, 0, a, b, g)
		V2_self_ = transform6para(V2_self, 0, 0, 0, a, b, g)
		V3_self_ = transform6para(V3_self, 0, 0, 0, a, b, g)
		V4_self_ = transform6para(V4_self, 0, 0, 0, a, b, g)
		# print ("V1, V2, V3 self_ is:", V1_self_, V2_self_, V3_self_)
		
		V1_ = np.array([V1_self_[0], V1_self_[1], V1_self_[2] + V1[2]])
		V2_ = np.array([V2_self_[0], V2_self_[1], V2_self_[2] + V2[2]])
		V3_ = np.array([V3_self_[0], V3_self_[1], V3_self_[2] + V3[2]])
		V4_ = np.array([V4_self_[0], V4_self_[1], V4_self_[2] + V4[2]])

		'''transform in camera xyz coordinate system. '''
		V1_ = transform6para(V1_, x, y, z, 0, 0, 0)
		V2_ = transform6para(V2_, x, y, z, 0, 0, 0)
		V3_ = transform6para(V3_, x, y, z, 0, 0, 0)
		V4_ = transform6para(V4_, x, y, z, 0, 0, 0)

		x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
		y_f_1 = y_f0 * V1_[1] / V1_[2] * (2000) / 135.8
		x_0f_1 = HALF_SZ + x_f_1
		y_0f_1 = HALF_SZ + y_f_1
		# print("x_f0 is: ", x_f0, "\nx_f_1 is: ", x_f_1)
		# print("y_f0 is: ", y_f0, "\ny_f_1 is: ", y_f_1)

		x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
		y_f_2 = y_f0 * V2_[1] / V2_[2] * (2000) / 135.8
		x_0f_2 = HALF_SZ + x_f_2
		y_0f_2 = HALF_SZ + y_f_2
		# print("x_f0 is: ", x_f0, "\nx_f_2 is: ", x_f_2)
		# print("y_f0 is: ", y_f0, "\ny_f_2 is: ", y_f_2)

		x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
		y_f_3 = y_f0 * V3_[1] / V3_[2] * (2000) / 135.8
		x_0f_3 = HALF_SZ + x_f_3
		y_0f_3 = HALF_SZ + y_f_3
		# print("x_f0 is: ", x_f0, "\nx_f_3 is: ", x_f_3)
		# print("y_f0 is: ", y_f0, "\ny_f_3 is: ", y_f_3)

		x_f_4 = x_f0 * V4_[0] / V4_[2] * (-2000) / 92.3
		y_f_4 = y_f0 * V4_[1] / V4_[2] * (2000) / 135.8
		x_0f_4 = HALF_SZ + x_f_4
		y_0f_4 = HALF_SZ + y_f_4

		# if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
			# y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
			# continue
		pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3], [x_0f_4, y_0f_4]])
		# print("pts1 is: ", pts1, "\npts2 is:", pts2)

		# M = cv2.getAffineTransform(pts1,pts2)
		M0 = cv2.getPerspectiveTransform(pts1,pts2)
		M = cv2.getPerspectiveTransform(pts2,pts1)
		print("M is ", M)
		print("M element is ", [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])


		# return 8 number parameter
		sample_matrixes.append([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])

		# dst = cv2.warpAffine(img, M, (width, height))
		img_resized = cv2.resize(img, (448, 448))
		dst_resized = cv2.warpPerspective(img_resized, M0, (448, 448))
		# dst_resized = cv2.resize(dst, (448, 448))
		
		cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
		cv2.imshow('camera', dst_resized)
		cv2.waitKey()
	return sample_matrixes


def random_sample_1000():

	sample_matrixes = []
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


	x_f0 = - (xmax - xmin) / 2 * SCALE
	y_f0 = - (ymax - ymin) / 2 * SCALE
	print(x_f0, y_f0)

	x_0f0_1 = HALF_SZ + x_f0
	y_0f0_1 = HALF_SZ + y_f0

	x_0f0_2 = HALF_SZ - x_f0
	y_0f0_2 = HALF_SZ + y_f0

	x_0f0_3 = HALF_SZ + x_f0
	y_0f0_3 = HALF_SZ - y_f0

	x_0f0_4 = HALF_SZ - x_f0
	y_0f0_4 = HALF_SZ - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3], [x_0f0_4,y_0f0_4]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0 * SCALE
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	x4, y4, z4 = 92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	V4 = np.array([x4, y4, z4])
	print(V1[0])
	print (V1, V2, V3, V4)


	# sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
	# 2000 ~ 
	max_a = math.pi * 2
	max_b = math.pi / 2
	max_g = math.pi / 2
	max_distance = 2600
	distance_step = 200

	# Get random Sample in Spherical coordinate system.

	# r_l = [500, 700, 900]
	# r_l = [900]
	# theta_l = [ \
	# 			0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# 	math.pi/12*5, \
	# 	math.pi/2 + 0.000000001,  \
	# 	math.pi/2 + math.pi/12, \
	# 	math.pi/2 + math.pi/6, \
	# 	math.pi/2 + math.pi/4, \
	# 	math.pi/2 + math.pi/3, \
	# 	math.pi/2 + math.pi/12*5, \
	# ]
	# phi_l = [ \
	# 	-math.pi/3, \
	# 	-math.pi/4, \
	# 	-math.pi/6, \
	# 	-math.pi/12, \
	# 	0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# ]


# Sample with 6 parameters.
	xl = [30, 100]
	yl = [30, 100]
	zl = [600, 900]
	abg = [	 				\
		# -math.pi/12*5, \
		-math.pi/3, \
		# -math.pi/4, \
		-math.pi/6, \
		# -math.pi/12, \
		0, \
		# math.pi/12, \
		math.pi/6, \
		# math.pi/4, \
		math.pi/3, \
		# math.pi/12*5  \
		]

	for x in xl:
		for y in yl:
			for z in zl:
				for a in abg:
					for b in abg:
						for g in abg:
							sample_6para_expand.append([x, y, z, a, b, g])

	for a in abg:
		for b in abg:
			for g in abg:
				print(a, b, g)
				for x in xl:
					for y in yl:
						for z in zl:
							sample_6para_expand.append([x, y, z, a, b, g])




	# for r in r_l:
	# 	for theta in theta_l:
	# 		for phi in phi_l:
	# 			z = r*math.sin(theta)*math.cos(phi)
	# 			x = -r*math.sin(theta)*math.sin(phi)
	# 			y = -r*math.cos(phi)
	# 			a = -math.atan(y/x)/2
	# 			b = phi/2
	# 			g = -math.atan(z/y)/2
	# 			sample_6para_expand.append([x, y, z, a, b, g])
	# 			print(r, theta, phi, x, y, z, a, b, g)





	for item in sample_6para_expand:
		x, y, z, a, b, g = item[0], item[1], item[2], item[3], item[4], item[5]

		print(x, y, z, a, b, g)

		# print ("V1, V2, V3 is:", V1, V2, V3)

		'''rotate in self coordinate system'''
		V1_self = np.array([V1[0], V1[1], 0])
		V2_self = np.array([V2[0], V2[1], 0])
		V3_self = np.array([V3[0], V3[1], 0])
		V4_self = np.array([V4[0], V4[1], 0])
		# print ("V1, V2, V3 self is:", V1_self, V2_self, V3_self)

		V1_self_ = transform6para(V1_self, 0, 0, 0, a, b, g)
		V2_self_ = transform6para(V2_self, 0, 0, 0, a, b, g)
		V3_self_ = transform6para(V3_self, 0, 0, 0, a, b, g)
		V4_self_ = transform6para(V4_self, 0, 0, 0, a, b, g)
		# print ("V1, V2, V3 self_ is:", V1_self_, V2_self_, V3_self_)
		
		V1_ = np.array([V1_self_[0], V1_self_[1], V1_self_[2] + V1[2]])
		V2_ = np.array([V2_self_[0], V2_self_[1], V2_self_[2] + V2[2]])
		V3_ = np.array([V3_self_[0], V3_self_[1], V3_self_[2] + V3[2]])
		V4_ = np.array([V4_self_[0], V4_self_[1], V4_self_[2] + V4[2]])

		'''transform in camera xyz coordinate system. '''
		V1_ = transform6para(V1_, x, y, z, 0, 0, 0)
		V2_ = transform6para(V2_, x, y, z, 0, 0, 0)
		V3_ = transform6para(V3_, x, y, z, 0, 0, 0)
		V4_ = transform6para(V4_, x, y, z, 0, 0, 0)

		x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
		y_f_1 = y_f0 * V1_[1] / V1_[2] * (2000) / 135.8
		x_0f_1 = HALF_SZ + x_f_1
		y_0f_1 = HALF_SZ + y_f_1
		# print("x_f0 is: ", x_f0, "\nx_f_1 is: ", x_f_1)
		# print("y_f0 is: ", y_f0, "\ny_f_1 is: ", y_f_1)

		x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
		y_f_2 = y_f0 * V2_[1] / V2_[2] * (2000) / 135.8
		x_0f_2 = HALF_SZ + x_f_2
		y_0f_2 = HALF_SZ + y_f_2
		# print("x_f0 is: ", x_f0, "\nx_f_2 is: ", x_f_2)
		# print("y_f0 is: ", y_f0, "\ny_f_2 is: ", y_f_2)

		x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
		y_f_3 = y_f0 * V3_[1] / V3_[2] * (2000) / 135.8
		x_0f_3 = HALF_SZ + x_f_3
		y_0f_3 = HALF_SZ + y_f_3
		# print("x_f0 is: ", x_f0, "\nx_f_3 is: ", x_f_3)
		# print("y_f0 is: ", y_f0, "\ny_f_3 is: ", y_f_3)

		x_f_4 = x_f0 * V4_[0] / V4_[2] * (-2000) / 92.3
		y_f_4 = y_f0 * V4_[1] / V4_[2] * (2000) / 135.8
		x_0f_4 = HALF_SZ + x_f_4
		y_0f_4 = HALF_SZ + y_f_4

		# if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
			# y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
			# continue
		pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3], [x_0f_4, y_0f_4]])
		# print("pts1 is: ", pts1, "\npts2 is:", pts2)

		# M = cv2.getAffineTransform(pts1,pts2)
		M0 = cv2.getPerspectiveTransform(pts1,pts2)
		M = cv2.getPerspectiveTransform(pts2,pts1)
		# print("M is ", M)
		# print("M element is ", [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])


		# return 8 number parameter
		sample_matrixes.append([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])

		# dst = cv2.warpAffine(img, M, (width, height))
		img_resized = cv2.resize(img, (448, 448))
		dst_resized = cv2.warpPerspective(img_resized, M0, (448, 448))
		# dst_resized = cv2.resize(dst, (448, 448))
		
		cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
		cv2.imshow('camera', dst_resized)
		cv2.waitKey()
	return sample_matrixes

def random_sample_125():

	sample_matrixes = []
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


	x_f0 = - (xmax - xmin) / 2 * SCALE
	y_f0 = - (ymax - ymin) / 2 * SCALE
	print(x_f0, y_f0)

	x_0f0_1 = HALF_SZ + x_f0
	y_0f0_1 = HALF_SZ + y_f0

	x_0f0_2 = HALF_SZ - x_f0
	y_0f0_2 = HALF_SZ + y_f0

	x_0f0_3 = HALF_SZ + x_f0
	y_0f0_3 = HALF_SZ - y_f0

	x_0f0_4 = HALF_SZ - x_f0
	y_0f0_4 = HALF_SZ - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3], [x_0f0_4,y_0f0_4]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0 * SCALE
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	x4, y4, z4 = 92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	V4 = np.array([x4, y4, z4])
	print(V1[0])
	print (V1, V2, V3, V4)


	# sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
	# 2000 ~ 
	max_a = math.pi * 2
	max_b = math.pi / 2
	max_g = math.pi / 2
	max_distance = 2600
	distance_step = 200

	# Get random Sample in Spherical coordinate system.

	# r_l = [500, 700, 900]
	# r_l = [900]
	# theta_l = [ \
	# 			0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# 	math.pi/12*5, \
	# 	math.pi/2 + 0.000000001,  \
	# 	math.pi/2 + math.pi/12, \
	# 	math.pi/2 + math.pi/6, \
	# 	math.pi/2 + math.pi/4, \
	# 	math.pi/2 + math.pi/3, \
	# 	math.pi/2 + math.pi/12*5, \
	# ]
	# phi_l = [ \
	# 	-math.pi/3, \
	# 	-math.pi/4, \
	# 	-math.pi/6, \
	# 	-math.pi/12, \
	# 	0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# ]


# Sample with 6 parameters.
	xl = [100]
	yl = [100]
	zl = [600, 900]
	abg = [	 				\
		# -math.pi/12*5, \
		-math.pi/3, \
		# -math.pi/4, \
		-math.pi/6, \
		# -math.pi/12, \
		0, \
		# math.pi/12, \
		math.pi/6, \
		# math.pi/4, \
		math.pi/3, \
		# math.pi/12*5  \
		]

	for x in xl:
		for y in yl:
			for z in zl:
				for a in abg:
					for b in abg:
						for g in abg:
							sample_6para_expand.append([x, y, z, a, b, g])

	for a in abg:
		for b in abg:
			for g in abg:
				print(a, b, g)
				for x in xl:
					for y in yl:
						for z in zl:
							sample_6para_expand.append([x, y, z, a, b, g])




	# for r in r_l:
	# 	for theta in theta_l:
	# 		for phi in phi_l:
	# 			z = r*math.sin(theta)*math.cos(phi)
	# 			x = -r*math.sin(theta)*math.sin(phi)
	# 			y = -r*math.cos(phi)
	# 			a = -math.atan(y/x)/2
	# 			b = phi/2
	# 			g = -math.atan(z/y)/2
	# 			sample_6para_expand.append([x, y, z, a, b, g])
	# 			print(r, theta, phi, x, y, z, a, b, g)





	for item in sample_6para_expand:
		x, y, z, a, b, g = item[0], item[1], item[2], item[3], item[4], item[5]

		print(x, y, z, a, b, g)

		# print ("V1, V2, V3 is:", V1, V2, V3)

		'''rotate in self coordinate system'''
		V1_self = np.array([V1[0], V1[1], 0])
		V2_self = np.array([V2[0], V2[1], 0])
		V3_self = np.array([V3[0], V3[1], 0])
		V4_self = np.array([V4[0], V4[1], 0])
		# print ("V1, V2, V3 self is:", V1_self, V2_self, V3_self)

		V1_self_ = transform6para(V1_self, 0, 0, 0, a, b, g)
		V2_self_ = transform6para(V2_self, 0, 0, 0, a, b, g)
		V3_self_ = transform6para(V3_self, 0, 0, 0, a, b, g)
		V4_self_ = transform6para(V4_self, 0, 0, 0, a, b, g)
		# print ("V1, V2, V3 self_ is:", V1_self_, V2_self_, V3_self_)
		
		V1_ = np.array([V1_self_[0], V1_self_[1], V1_self_[2] + V1[2]])
		V2_ = np.array([V2_self_[0], V2_self_[1], V2_self_[2] + V2[2]])
		V3_ = np.array([V3_self_[0], V3_self_[1], V3_self_[2] + V3[2]])
		V4_ = np.array([V4_self_[0], V4_self_[1], V4_self_[2] + V4[2]])

		'''transform in camera xyz coordinate system. '''
		V1_ = transform6para(V1_, x, y, z, 0, 0, 0)
		V2_ = transform6para(V2_, x, y, z, 0, 0, 0)
		V3_ = transform6para(V3_, x, y, z, 0, 0, 0)
		V4_ = transform6para(V4_, x, y, z, 0, 0, 0)

		x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
		y_f_1 = y_f0 * V1_[1] / V1_[2] * (2000) / 135.8
		x_0f_1 = HALF_SZ + x_f_1
		y_0f_1 = HALF_SZ + y_f_1
		# print("x_f0 is: ", x_f0, "\nx_f_1 is: ", x_f_1)
		# print("y_f0 is: ", y_f0, "\ny_f_1 is: ", y_f_1)

		x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
		y_f_2 = y_f0 * V2_[1] / V2_[2] * (2000) / 135.8
		x_0f_2 = HALF_SZ + x_f_2
		y_0f_2 = HALF_SZ + y_f_2
		# print("x_f0 is: ", x_f0, "\nx_f_2 is: ", x_f_2)
		# print("y_f0 is: ", y_f0, "\ny_f_2 is: ", y_f_2)

		x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
		y_f_3 = y_f0 * V3_[1] / V3_[2] * (2000) / 135.8
		x_0f_3 = HALF_SZ + x_f_3
		y_0f_3 = HALF_SZ + y_f_3
		# print("x_f0 is: ", x_f0, "\nx_f_3 is: ", x_f_3)
		# print("y_f0 is: ", y_f0, "\ny_f_3 is: ", y_f_3)

		x_f_4 = x_f0 * V4_[0] / V4_[2] * (-2000) / 92.3
		y_f_4 = y_f0 * V4_[1] / V4_[2] * (2000) / 135.8
		x_0f_4 = HALF_SZ + x_f_4
		y_0f_4 = HALF_SZ + y_f_4

		# if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
			# y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
			# continue
		pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3], [x_0f_4, y_0f_4]])
		# print("pts1 is: ", pts1, "\npts2 is:", pts2)

		# M = cv2.getAffineTransform(pts1,pts2)
		M0 = cv2.getPerspectiveTransform(pts1,pts2)
		M = cv2.getPerspectiveTransform(pts2,pts1)
		# print("M is ", M)
		# print("M element is ", [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])


		# return 8 number parameter
		sample_matrixes.append([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])

		# dst = cv2.warpAffine(img, M, (width, height))
		img_resized = cv2.resize(img, (448, 448))
		dst_resized = cv2.warpPerspective(img_resized, M0, (448, 448))
		# dst_resized = cv2.resize(dst, (448, 448))
		
		cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
		cv2.imshow('camera', dst_resized)
		cv2.waitKey()
	return sample_matrixes

def random_sample_33():

	sample_matrixes = []
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


	x_f0 = - (xmax - xmin) / 2 * SCALE
	y_f0 = - (ymax - ymin) / 2 * SCALE
	print(x_f0, y_f0)

	x_0f0_1 = HALF_SZ + x_f0
	y_0f0_1 = HALF_SZ + y_f0

	x_0f0_2 = HALF_SZ - x_f0
	y_0f0_2 = HALF_SZ + y_f0

	x_0f0_3 = HALF_SZ + x_f0
	y_0f0_3 = HALF_SZ - y_f0

	x_0f0_4 = HALF_SZ - x_f0
	y_0f0_4 = HALF_SZ - y_f0


	pts1 = np.float32([[x_0f0_1,y_0f0_1],[x_0f0_2,y_0f0_2],[x_0f0_3,y_0f0_3], [x_0f0_4,y_0f0_4]])
	# print(pts1)

	# get camera focal length f (unit pixel) with A4 paper parameter.
	f = (-2000) / 92.3 * x_f0 * SCALE
	print ("estimate focal length: ", f, " pixel")

	# 3 points of A4 paper in camera coordinate system.
	x1, y1, z1 = -92.3, 135.8, 2000
	x2, y2, z2 = 92.3, 135.8, 2000
	x3, y3, z3 = -92.3, -135.8, 2000
	x4, y4, z4 = 92.3, -135.8, 2000
	print(x1, y2, z3)

	V1 = np.array([x1, y1, z1])
	V2 = np.array([x2, y2, z2])
	V3 = np.array([x3, y3, z3])
	V4 = np.array([x4, y4, z4])
	print(V1[0])
	print (V1, V2, V3, V4)


	# sample x,y,z  a,b,g: 0~2*pi, -pi/2 ~ pi/2
	# 2000 ~ 
	max_a = math.pi * 2
	max_b = math.pi / 2
	max_g = math.pi / 2
	max_distance = 2600
	distance_step = 200

	# Get random Sample in Spherical coordinate system.

	# r_l = [500, 700, 900]
	# r_l = [900]
	# theta_l = [ \
	# 			0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# 	math.pi/12*5, \
	# 	math.pi/2 + 0.000000001,  \
	# 	math.pi/2 + math.pi/12, \
	# 	math.pi/2 + math.pi/6, \
	# 	math.pi/2 + math.pi/4, \
	# 	math.pi/2 + math.pi/3, \
	# 	math.pi/2 + math.pi/12*5, \
	# ]
	# phi_l = [ \
	# 	-math.pi/3, \
	# 	-math.pi/4, \
	# 	-math.pi/6, \
	# 	-math.pi/12, \
	# 	0 + 0.000000001, \
	# 	math.pi/12, \
	# 	math.pi/6, \
	# 	math.pi/4, \
	# 	math.pi/3, \
	# ]
# Saple with 6 parameters.
	xl = [100]
	yl = [100]
	zl = [900]
	abg = [	 				\
		# -math.pi/12*5, \
		# -math.pi/3, \
		# -math.pi/4, \
		-math.pi/60*3, \
		-math.pi/60*2, \
		-math.pi/60, \
		# -math.pi/12, \
		0, \
		# math.pi/12, \
		math.pi/60, \
		math.pi/60*2, \
		math.pi/60*3,
		# math.pi/4, \
		# math.pi/3, \
		# math.pi/12*5  \
		]


	# for x in xl:
	# 	for y in yl:
	# 		for z in zl:
	# 			for a in abg:
	# 				for b in abg:
	# 					for g in abg:
	# 						sample_6para_expand.append([x, y, z, a, b, g])

	# for a in abg:
	# 	# print(a, b, g)
	# 	for x in xl:
	# 		for y in yl:
	# 			for z in zl:
	# 				sample_6para_expand.append([x, y, z, a, 0, 0])
	# for b in abg:
	# 	# print(a, b, g)
	# 	for x in xl:
	# 		for y in yl:
	# 			for z in zl:
	# 				sample_6para_expand.append([x, y, z, 0, b, 0])
	# for g in abg:
	# 	# print(a, b, g)
	# 	for x in xl:
	# 		for y in yl:
	# 			for z in zl:
	# 				sample_6para_expand.append([x, y, z, 0, 0, g])

	for i in range(-10, 10, 1):
		sample_6para_expand.append([100, 100, 900, math.pi/60*i, 0, 0])
	for i in range(-10, 10, 1):
		sample_6para_expand.append([100, 100, 900, 0, math.pi/60*i, 0])
	for i in range(-10, 10, 1):
		sample_6para_expand.append([100, 100, 900, 0, 0, math.pi/60*i])
	
	for item in sample_6para_expand:
		x, y, z, a, b, g = item[0], item[1], item[2], item[3], item[4], item[5]

		print(x, y, z, a, b, g)

		# print ("V1, V2, V3 is:", V1, V2, V3)

		'''rotate in self coordinate system'''
		V1_self = np.array([V1[0], V1[1], 0])
		V2_self = np.array([V2[0], V2[1], 0])
		V3_self = np.array([V3[0], V3[1], 0])
		V4_self = np.array([V4[0], V4[1], 0])
		# print ("V1, V2, V3 self is:", V1_self, V2_self, V3_self)

		V1_self_ = transform6para(V1_self, 0, 0, 0, a, b, g)
		V2_self_ = transform6para(V2_self, 0, 0, 0, a, b, g)
		V3_self_ = transform6para(V3_self, 0, 0, 0, a, b, g)
		V4_self_ = transform6para(V4_self, 0, 0, 0, a, b, g)
		# print ("V1, V2, V3 self_ is:", V1_self_, V2_self_, V3_self_)
		
		V1_ = np.array([V1_self_[0], V1_self_[1], V1_self_[2] + V1[2]])
		V2_ = np.array([V2_self_[0], V2_self_[1], V2_self_[2] + V2[2]])
		V3_ = np.array([V3_self_[0], V3_self_[1], V3_self_[2] + V3[2]])
		V4_ = np.array([V4_self_[0], V4_self_[1], V4_self_[2] + V4[2]])

		'''transform in camera xyz coordinate system. '''
		V1_ = transform6para(V1_, x, y, z, 0, 0, 0)
		V2_ = transform6para(V2_, x, y, z, 0, 0, 0)
		V3_ = transform6para(V3_, x, y, z, 0, 0, 0)
		V4_ = transform6para(V4_, x, y, z, 0, 0, 0)

		x_f_1 = x_f0 * V1_[0] / V1_[2] * (-2000) / 92.3
		y_f_1 = y_f0 * V1_[1] / V1_[2] * (2000) / 135.8
		x_0f_1 = HALF_SZ + x_f_1
		y_0f_1 = HALF_SZ + y_f_1
		# print("x_f0 is: ", x_f0, "\nx_f_1 is: ", x_f_1)
		# print("y_f0 is: ", y_f0, "\ny_f_1 is: ", y_f_1)

		x_f_2 = x_f0 * V2_[0] / V2_[2] * (-2000) / 92.3
		y_f_2 = y_f0 * V2_[1] / V2_[2] * (2000) / 135.8
		x_0f_2 = HALF_SZ + x_f_2
		y_0f_2 = HALF_SZ + y_f_2
		# print("x_f0 is: ", x_f0, "\nx_f_2 is: ", x_f_2)
		# print("y_f0 is: ", y_f0, "\ny_f_2 is: ", y_f_2)

		x_f_3 = x_f0 * V3_[0] / V3_[2] * (-2000) / 92.3
		y_f_3 = y_f0 * V3_[1] / V3_[2] * (2000) / 135.8
		x_0f_3 = HALF_SZ + x_f_3
		y_0f_3 = HALF_SZ + y_f_3
		# print("x_f0 is: ", x_f0, "\nx_f_3 is: ", x_f_3)
		# print("y_f0 is: ", y_f0, "\ny_f_3 is: ", y_f_3)

		x_f_4 = x_f0 * V4_[0] / V4_[2] * (-2000) / 92.3
		y_f_4 = y_f0 * V4_[1] / V4_[2] * (2000) / 135.8
		x_0f_4 = HALF_SZ + x_f_4
		y_0f_4 = HALF_SZ + y_f_4

		# if(x_0f_1 < 0 or x_0f_1 > 3024 or y_0f_1 < 0 or y_0f_1 > 3024 or x_0f_2 < 0 or x_0f_2 > 3024 or \
			# y_0f_2 < 0 or y_0f_2 > 3024 or x_0f_3 < 0 or x_0f_3 > 3024 or y_0f_3 < 0 or y_0f_3 > 3024):
			# continue
		pts2 = np.float32([[x_0f_1, y_0f_1],[x_0f_2, y_0f_2],[x_0f_3, y_0f_3], [x_0f_4, y_0f_4]])
		# print("pts1 is: ", pts1, "\npts2 is:", pts2)

		# M = cv2.getAffineTransform(pts1,pts2)
		M0 = cv2.getPerspectiveTransform(pts1,pts2)
		M = cv2.getPerspectiveTransform(pts2,pts1)
		# print("M is ", M)
		# print("M element is ", [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])


		# return 8 number parameter
		sample_matrixes.append([M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1]])

		# dst = cv2.warpAffine(img, M, (width, height))
		img_resized = cv2.resize(img, (448, 448))
		dst_resized = cv2.warpPerspective(img_resized, M0, (448, 448))
		# dst_resized = cv2.resize(dst, (448, 448))
		
# 		cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
# 		cv2.imshow('camera', dst_resized)
# 		cv2.waitKey()

	return sample_matrixes

if __name__=='__main__':
	#test_transform()
	# target_sample()
	# random_sample_1000()
	# random_sample_125()
	random_sample_33()
