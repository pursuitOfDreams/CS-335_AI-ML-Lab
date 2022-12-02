from utils import *
import numpy as np
#import pandas as pd

############ QUESTION 3 ##############
class KR:
	def __init__(self, x,y,b=1):
		self.x = x
		self.y = y
		self.b = b
	
	def gaussian_kernel(self, z):
		'''
		Implement gaussian kernel
		''' 	
		a = (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)
		#print(a)
		return a
	
	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_train, ) 
		'''
		y_test = []
		for X in x_test:
			kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
			weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
			y_test.append(np.dot(weights, self.y)/len(self.x))
		return np.array(y_test)
		
def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()
	
	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	#print(x_train)
	print("Loss = " ,find_loss(y_test, y_predicted))

q3()
	
