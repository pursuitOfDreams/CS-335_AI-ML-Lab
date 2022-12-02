
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os



############################ Q3 ################################

def GaussianFilter(x, w,stride):
    """
    A naive implementation of gradient filter convolution.
    The input consists of N data points,height H and
    width W. We convolve each input with F different filters and has height HH and width WW.
    Input:
    - x: Input data of shape (N, H, W)
    - w: Filter weights of shape (F, HH, WW)
    - stride: The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    Return:
   - out: Output data, of shape (N, F, H, W)
    """
    ##Note:if the mean value from a filter is float,perform ceil operation i.e.,29.2--->30
    #################### Enter Your Code Here
    N, H, W = x.shape
    F, HH, WW = w.shape
    delta_h, delta_w = int(HH/2), int(WW/2)

    out = np.expand_dims(x,axis=1)
    out = np.repeat(out, F, axis=1)
    
    for h in range(delta_h, H-delta_h, stride):
        for w1 in range(delta_w, W-delta_w, stride):
            for f in range(F):
                out[:,f,h,w1] = np.ceil(np.sum(np.sum(w[f]*x[:,h-delta_h:h+delta_h+1,w1-delta_w:w1+delta_w+1],axis = 2), axis = 1)/np.sum(np.sum(w, axis = 2), axis = 1))
    return out

x_shape = (1, 6,6)
w_shape = (1,3,3)
x = np.array([[15,20,25,25,15,10],[20,15,50,30,20,15],[20,50,55,60,30,20],[20,15,65,30,15,30],[15,20,30,20,25,30],[20,25,15,20,10,15]]).reshape(x_shape)
w = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]).reshape(w_shape)
stride=1
out = GaussianFilter(x, w, stride)
# print(out)
# correct_out=np.array([[[[15, 20, 25, 25, 15, 10],
#          [20, 29, 38, 35, 24, 15],
#          [20, 36, 48, 43, 29, 20],
#          [20, 31, 42, 37, 27, 30],
#          [15, 24, 29, 25, 22, 30],
#          [20, 25, 15, 20, 10, 15]]]])

# print(out==correct_out)