import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm


############################ Q3 ################################

def q3(x, w, b, conv_param):
    """
    A naive implementation of convolution.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N, C, H, W  = x.shape
    F, C, HH, WW = w.shape

    padding, stride = conv_param['pad'], conv_param['stride']
    out_h = int((H + 2 * padding - HH) / stride)+1
    out_w = int((W + 2 * padding - WW) / stride)+1
    mask = np.zeros(( N, C, H+2*padding, W+2*padding))
    mask[:,:,padding:-padding,padding:-padding]= x
    pad_img = mask
    # pad_img= np.pad(X, ((0,0),(0,0),(1,1),(1,1)),mode='constant') 
    # out = np.array() #Output of the convolution


    out = np.zeros((N, F, out_h, out_w))
    for n in range(N):
      for x in range(0,out_h):
        for y in range(0,out_w):
          for f in range(0,F):
            conv = pad_img[n,:,x*stride:x*stride+HH,y*stride:y*stride+WW]*w[f,:,:,:]
            sum1 = np.sum(conv)+b[f]
            out[n,f,x,y] = sum1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def gram(x):
  ######START: TO CODE########
  #Returns the gram matrix
  x1 = x.reshape(-1,x.shape[2])
  return x1.T@x1
  ######END: TO CODE########


def relative_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))