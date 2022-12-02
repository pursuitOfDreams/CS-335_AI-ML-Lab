from utils import *
import numpy as np


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
		
	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_train, ) 
		'''
		
def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()
	
	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	print("Loss = " ,find_loss(y_test, y_predicted))

	
