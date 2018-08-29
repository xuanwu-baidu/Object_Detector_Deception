import numpy as np
# import tensorflow as tf
# import cv2
import math

import eulerangles as eu
# read a image with corp

# transform it.




def transform(V, Mt, Mr):
	return np.dot(Mr, V + Mt)

def transform6para(V, transx = 0, transy = 0, transz = 0, rotz = 0, roty = 0, rotx = 0):
	Mt = [transx, transy, transz]
	Mr = eu.euler2mat(rotz, roty, rotx)
	return np.dot(Mr, V +  Mt)
# write the 6 num -> Mx as a function and test it.

if __name__=='__main__':

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