import numpy as np
from matplotlib import pyplot as plt
import math
import pickle as pkl

np.random.seed(0)

def preprocessing(X):
    """
    Implement Normalization for input image features

    Args:
    X : numpy array of shape (n_samples, n_features)
    
    Returns:
    X_out: numpy array of shape (n_samples, n_features) after normalization
    """
    X_out = None
    
    ## TODO
    X_out = (X-np.min(X,axis=0))/(np.maximum(np.max(X,axis=0)-np.min(X,axis=0), 1e-6))
    # X_mean = np.mean(X,axis=0)
    # X_std = np.std(X,axis=0)
    # X_out = (X-X_mean)/(X_std+1e-6)
    ## END TODO

    assert X_out.shape == X.shape

    return X_out

def split_data(X, Y, train_ratio=0.8):
    '''
    Split data into train and validation sets
    The first floor(train_ratio*n_sample) samples form the train set
    and the remaining the validation set

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    train_ratio - fraction of samples to be used as training data

    Returns:
    X_train, Y_train, X_val, Y_val
    '''
    # Try Normalization and scaling and store it in X_transformed
    X_transformed = preprocessing(X)

    ## TODO
    
    ## END TODO

    assert X_transformed.shape == X.shape

    num_samples = len(X)
    indices = np.arange(num_samples)
    num_train_samples = math.floor(num_samples * train_ratio)
    train_indices = np.random.choice(indices, num_train_samples, replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[train_indices], X_transformed[val_indices], Y[val_indices]
  
    return X_train, Y_train, X_val, Y_val

class FlattenLayer:
    '''
    This class converts a multi-dimensional into 1-d vector
    '''
    def __init__(self, input_shape):
        '''
        Args:
        input_shape : Original shape, tuple of ints
        '''
        self.input_shape = input_shape

    def forward(self, input):
        '''
        Converts a multi-dimensional into 1-d vector
        Args:
          input : training data, numpy array of shape (n_samples , self.input_shape)

        Returns:
          input: training data, numpy array of shape (n_samples , -1)
        '''
        ## TODO
        # assuming that the input is of shape (-1)
        input = input.reshape((input.shape[0],-1))
        #Modify the return statement to return flattened input
        return input
        ## END TODO
        
    
    def backward(self, output_error, learning_rate):
        '''
        Converts back the passed array to original dimention 
        Args:
        output_error :  numpy array 
        learning_rate: float

        Returns:
        output_error: A reshaped numpy array to allow backward pass
        '''
        ## TODO
        # output_error = output_error.reshape(self.input_shape)
        #Modify the return statement to return reshaped array
        return output_error
        ## END TODO
        
        
class FCLayer:
    '''
    Implements a fully connected layer  
    '''
    def __init__(self, input_size, output_size):
        '''
        Args:
         input_size : Input shape, int
         output_size: Output shape, int 
        '''
        self.input_size = input_size
        self.output_size = output_size
        ## TODO 
        xw = np.sqrt(6/(self.input_size+self.output_size))
        xb = np.sqrt(6/(1+self.output_size))
        self.weights = np.random.randn(input_size, output_size)*(np.sqrt(2/(self.input_size+self.output_size))) #initilaise weights for this layer
        self.bias = np.random.random((1, output_size))-0.5  #initilaise bias for this layer
        # self.bias = np.zeros((1,output_size))
        ## END TODO

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        Args:
          input : training data, numpy array of shape (n_samples , self.input_size)

        Returns:
           numpy array of shape (n_samples , self.output_size)
        '''
        ## TODO
        self.input = input.copy()
        self.out = input@self.weights+self.bias
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        # output shape (1, output_size)
        return self.out
        ## END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error 
          :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        #update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        #Modify the return statement to return numpy array resulting from backward pass
        return input_error
        ## END TODO
        
        
class ActivationLayer:
    '''
    Implements a Activation layer which applies activation function on the inputs. 
    '''
    def __init__(self, activation, activation_prime):
        '''
          Args:
          activation : Name of the activation function (sigmoid,tanh or relu)
          activation_prime: Name of the corresponding function to be used during backpropagation (sigmoid_prime,tanh_prime or relu_prime)
        '''
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        '''
        Applies the activation function 
        Args:
          input : numpy array on which activation function is to be applied

        Returns:
           numpy array output from the activation function
        '''
        ## TODO
        self.input = input.copy()
        self.output = self.activation(self.input)
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return self.output
        ## END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        #Modify the return statement to return numpy array resulting from backward pass
        return self.activation_prime(self.input)*output_error
        ## END TODO
        
        

class SoftmaxLayer:
    '''
      Implements a Softmax layer which applies softmax function on the inputs. 
    '''
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        '''
        Applies the softmax function 
        Args:
          input : numpy array on which softmax function is to be applied

        Returns:
           numpy array output from the softmax function
        '''
        ## TODO
        self.input = input.copy()
        m = np.max(input,axis =1)
        self.output = np.exp(input-m)/np.sum(np.exp(input-m), axis = 1, keepdims=True)
        # print(np.sum(np.exp(input-m), axis = 0))
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        return self.output
        ## END TODO
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        
        n = self.output.shape[0]
        d = self.output.shape[1]
        input_error = -self.output.T@self.output
        input_error = (np.ones((d,d))-np.eye(d))*input_error+ np.eye(d)*self.output*(1-self.output)
        input_error = output_error@input_error

        # input_error = output_error@np.squeeze(self.output[:,None,:]*(np.eye(d)[None,:,:]-self.output[:,None,:]))
        #Modify the return statement to return numpy array resulting from backward pass
        return input_error
        ## END TODO
        
        
def sigmoid(x):
    '''
    Sigmoid function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying simoid function
    '''
    ## TODO
    x1= np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
    #Modify the return statement to return numpy array resulting from backward pass
    return x1
    ## END TODO

def sigmoid_prime(x):
    '''
     Implements derivative of Sigmoid function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Sigmoid function
    '''
    ## TODO
    s = sigmoid(x)
    #Modify the return statement to return numpy array resulting from backward pass
    return s*(1-s)
    ## END TODO

def tanh(x):
    '''
    Tanh function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying tanh function
    '''
    ## TODO
    x1= np.tanh(x)
    #Modify the return statement to return numpy array resulting from backward pass
    return x1
    ## END TODO

def tanh_prime(x):
    '''
     Implements derivative of Tanh function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Tanh function
    '''
    a = np.tanh(x)
    x1 = 1-a**2
    ## TODO

    #Modify the return statement to return numpy array resulting from backward pass
    return x1
    ## END TODO

def relu(x):
    '''
    ReLU function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying ReLU function
    '''
    ## TODO
    x1 = np.maximum(0,x)
    #Modify the return statement to return numpy array resulting from backward pass
    return x1
    ## END TODO

def relu_prime(x):
    '''
     Implements derivative of ReLU function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of ReLU function
    '''
    ## TODO
    x1 = x>=0
    x1 = x1.astype(int)
    #Modify the return statement to return numpy array resulting from backward pass
    return x1
    ## END TODO
    
def cross_entropy(y_true, y_pred):
    '''
    Cross entropy loss 
    Args:
        y_true :  Ground truth labels, numpy array 
        y_pred :  Predicted labels, numpy array 
    Returns:
       loss : float
    '''
    ## TODO
    y = np.zeros(y_pred.shape)
    y[np.arange(0,y_pred.shape[0]), y_true] =1.0
    loss = -(np.sum(y*np.log((y_pred+1e-6)), axis=1))
    #Modify the return statement to return numpy array resulting from backward pass
    return loss
    ## END TODO

def cross_entropy_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of cross entropy function
    '''
    ## TODO
    y = np.zeros(y_pred.shape)
    y[np.arange(0,y_pred.shape[0]), y_true] =1.0
    dz = -y*(1/(y_pred+1e-6))
    #Modify the return statement to return numpy array resulting from backward pass
    return dz
    ## END TODO
    
    
def fit(X_train, Y_train, X_test, Y_test, dataset_name):

    '''
    Create and trains a feedforward network

    Do not forget to save the final model/weights of the feed forward network to a file. Use these weights in the `predict` function 
    Args:
        X_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        Y_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        dataset_name -- name of the dataset (flowers or mnist)
    
    '''
     
    #Note that this just a template to help you create your own feed forward network 
    ## TODO

    #define your network
    #This example network below would work only for mnist
    #you will need separtae networks for the two datasets
    if dataset_name =="mnist":
        network = [
            FlattenLayer(input_shape=(28, 28)),
            FCLayer(28 * 28, 32),
            ActivationLayer(sigmoid, sigmoid_prime),
            FCLayer(32, 10),
            SoftmaxLayer(10)
        ] # This creates feed forward 
    elif dataset_name == "flowers":
        network = [
            FCLayer(2048,48),
            ActivationLayer(sigmoid, sigmoid_prime),
            FCLayer(48,5),
            SoftmaxLayer(5)
        ]


    # Choose appropriate learning rate and no. of epoch
    epochs = 60 if dataset_name=="mnist" else 23
    learning_rate = 0.01
    batch_size = 128

    # x_train  = np.array([X_train[i:(i+batch_size)] for i in range(0, len(X_train)-batch_size, batch_size)]+[X_train[int(len(X_train)/batch_size)*batch_size:]])
    # y_train  = np.array([Y_train[i:(i+batch_size)] for i in range(0, len(X_train)-batch_size, batch_size)]+[Y_train[int(len(X_train)/batch_size)*batch_size:]])
    # Change training loop as you see fit
    for epoch in range(epochs):
        error = 0
        Y_pred = np.zeros(X_train.shape[0],)
        for i, (x, y_true) in enumerate(zip(X_train, Y_train)):
            # forward
            output = np.reshape(x,(1,-1)) 
            for layer in network:
                output = layer.forward(output)
            
            Y_pred[i]=np.argmax(output)

            # error (display purpose only)
            error1 = cross_entropy(y_true, output)
            # print(error1)
            error += error1
            # backward
            output_error = cross_entropy_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)

        train_accuracy = np.sum(Y_pred==Y_train)/len(Y_train)

        Y_pred = np.zeros(X_test.shape[0],)

        ## TODO
        for i,x in enumerate(X_test):
            # forward
            output = np.reshape(x,(1,-1)) 
            for layer in network:
                output = layer.forward(output)

            Y_pred[i] = np.argmax(output)
    
        val_accuracy = np.sum(Y_pred==Y_test)/len(Y_test)
        

        if epoch%10==0 and epoch!=0:
            learning_rate*=0.5
        error /= len(X_train)
        print(40*"-",f"EPOCH {epoch}",40*"-")
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))
        print("training accuracy", train_accuracy)
        print("validation accuracy", val_accuracy)
    #Save you model/weights as ./models/{dataset_name}_model.pkl

    dic = {}
    dic["model"] = network
    with open(f"./models/{dataset_name}_model.pkl",'wb') as f:
        pkl.dump(network,f)
    
    ## END TODO
    
def predict(X_test, dataset_name):
    """

    X_test -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.

    This is the function that we will call from the auto grader. 

    This function should only perform inference, please donot train your models here.

    Steps to be done here:
    1. Load your trained model/weights from ./models/{dataset_name}_model.pkl
    2. Ensure that you read model/weights using only the libraries we have given above.
    3. In case you have saved weights only in your file, itialize your model with your trained weights.
    4. Compute the predicted labels and return it

    Return:
    Y_test - nparray of shape (num_test,)
    """
    Y_pred = np.zeros(X_test.shape[0],)

    with open(f"./models/{dataset_name}_model.pkl",'rb') as f:
        network = pkl.load(f)

    ## TODO
    for i,x in enumerate(X_test):
        # forward
        output = np.reshape(x,(1,-1)) 
        for layer in network:
            output = layer.forward(output)

        Y_pred[i] = np.argmax(output)

    ## END TODO
    assert Y_pred.shape == (X_test.shape[0],) and type(Y_pred) == type(X_test), "Check what you return"
    return Y_pred
    
# def get_accuracy(X_test, Y_test):

"""
Loading data and training models
"""
if __name__ == "__main__":    
    # dataset = "mnist" 
    # with open(f"./data/{dataset}_train.pkl", "rb") as file:
    #     train_mnist = pkl.load(file)
    #     X_train, Y_train, X_test, Y_test = split_data(train_mnist[0], train_mnist[1])
    #     print(f"train_x -- {train_mnist[0].shape}; train_y -- {train_mnist[1].shape}")
    #     fit(X_train, Y_train, X_test, Y_test,'mnist')
    #     # Y_pred = predict(X_test, dataset)
    #     # accuracy = np.sum(Y_pred==Y_test)/len(Y_test)
    #     # print("accuracy", accuracy)
    
    dataset = "flowers"
    with open(f"./data/{dataset}_train.pkl", "rb") as file:
        train_flowers = pkl.load(file)
        X_train, Y_train, X_test, Y_test = split_data(train_flowers[0], train_flowers[1])
        print(f"train_x -- {train_flowers[0].shape}; train_y -- {train_flowers[1].shape}")
        fit(X_train, Y_train, X_test, Y_test,'flowers')
        # Y_pred = predict(X_test, dataset)
        # accuracy = np.sum(Y_pred==Y_test)/len(Y_test)

