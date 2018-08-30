#import for transformation
import numpy as np
import math
import eulerangles as eu

#import for image generation
import cv2
import xmltodict
# read a image with corp

# transform it.

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

def generate_photo():
	img = cv2.imread("test/Darren.jpg")
	height, width, channels = img.shape
	# rows, cols = image.shape
	print(height, width, channels)

	f = open("test/Darren.xml")
	dic = xmltodict.parse(f.read())

	xmin = 0
	ymin = 0
	xmax = 0
	ymax = 0
	for _object in dic['annotation']['object']:
		xmin = int(_object['bndbox']['xmin'])
		ymin = int(_object['bndbox']['ymin'])
		xmax = int(_object['bndbox']['xmax'])
		ymax = int(_object['bndbox']['ymax'])

		break
	print(xmin,ymin,xmax,ymax)


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
	generate_photo()

