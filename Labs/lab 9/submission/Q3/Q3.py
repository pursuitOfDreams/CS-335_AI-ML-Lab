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
		out = (1/(np.sqrt(2*np.pi)))*(np.exp(-(z**2)/2))
		return out
	
	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_train, ) 
		'''

		w = self.gaussian_kernel((self.x[:,None]-x_test[None,:])/self.b)
		w = w/np.sum(w,axis = 0)*self.x.shape[0]
		w = w.squeeze()
		Y = self.y.reshape(self.y.shape[0],1)
		return np.sum(w*Y,axis=0)/self.x.shape[0]
		
		
def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()
	
	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	print("Loss = " ,find_loss(y_test, y_predicted))
	
q3()
