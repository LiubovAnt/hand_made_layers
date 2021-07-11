#coding=utf-8

# Library with layers for Technotrack task #1
import numpy as np
## Layes
class Linear:
    def __init__(self, input_size, output_size, no_b=False):
        '''
        Creates weights and biases for linear layer from N(0, 0.01).
        Dimention of inputs is *input_size*, of output: *output_size*.
        no_b=True - do not use interception in prediction and backward (y = w*X)
        '''
        #### YOUR CODE HERE
        # Инициализация собственных значений весов
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=[input_size, output_size])
        self.no_b = no_b
        if not no_b:
            self.biases = np.random.normal(loc=0.0, scale=0.01, size=[1, output_size])
        #print (self.weights)
        #pass
    
    # N - batch_size
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        # 
        # Сохраняем для обратного прохода
        self.X = X
        output = X @ self.weights
        if not self.no_b:
            output = output + self.biases
        return output
        #pass

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        self.dLdW = self.X.transpose()@dLdy/self.X.shape[0]
        dLdX = dLdy@self.weights.transpose()
        if not self.no_b:
            self.dLdb = np.sum(dLdy)/dLdy.shape[0]
        return dLdX
        #pass

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        #### YOUR CODE HERE
        self.weights = self.weights - learning_rate * self.dLdW
        #print ("w", self.weights)
        if not self.no_b:
            self.biases = self.biases - learning_rate * self.dLdb
            #print ("b", self.biases)
 


## Activations
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        #### YOUR CODE HERE
        #### Apply layer to input
        sigma = 1/(1+np.exp(-X))
        self.sigma = sigma
        return sigma

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        #### YOUR CODE HERE
        dLdx = dLdy * (self.sigma*(1-self.sigma))
        return dLdx
        # pass

    def step(self, learning_rate):
        pass

class ELU:
    def __init__(self, alpha):
        #### YOUR CODE HERE
        self.alpha = alpha
        #pass

    def forward(self, X):
        #### YOUR CODE HERE
        self.X = X
        return np.where(self.X < 0, self.alpha * (np.exp(X) - 1), X)
        #pass

    def backward(self, dLdy):
        #### YOUR CODE HERE
        return dLdy*np.where(self.X < 0, self.alpha * np.exp(self.X), 1)
        #pass

    def step(self, learning_rate):
        pass


class ReLU:
    def __init__(self, a):
        #### YOUR CODE HERE
        self.a = a
        #pass

    def forward(self, X):
        #### YOUR CODE HERE
        self.X = X
        return np.where(X < 0, self.a * X, X)
        #pass
      
    def backward(self, dLdy):
        #### YOUR CODE HERE
        return dLdy*np.where(self.X < 0, self.a, 1)
        #pass

    def step(self, learning_rate):
        pass


class Tanh:
    def forward(self, X):
        #### YOUR CODE HERE
        self.tanh = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
        return self.tanh
        #pass

    def backward(self, dLdy):
        #### YOUR CODE HERE
        return dLdy*(1-self.tanh**2)
        #pass

    def step(self, learning_rate):
        pass


## Final layers, loss functions
class SoftMax_NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass

    def forward(self, X):
        '''
        Returns SoftMax for all X (matrix with size X.shape, containing in lines probabilities of each class)
        '''
        #self.X = X
        self.p = np.exp(X)/np.expand_dims(np.sum(np.exp(X),axis=1), axis=1)
        #print ("p", self.p)
        return self.p
        #### YOUR CODE HERE
        #pass

    # y - true labels. Calculates dL/dy, returns dL/dX
    def get_loss(self, y, numb):
        dLdy= np.zeros((y.size, numb))
        for i in range(y.size):
            dLdy[i][y[i]]=1
        #print (dLdy)
        eps = 10**(-9)
        Loss =  - dLdy * np.log(self.p+eps) - (1 - dLdy) * np.log(1 - self.p+eps)
        #print('Loss:', np.sum(Loss))
        return np.sum(Loss)/dLdy.shape[0]
        
    def backward(self, y):
        dLdy= np.zeros((y.size, self.p.shape[1]))
        for i in range(y.size):
            dLdy[i][y[i]]=1
        #eps = 10**(-9)
        #Loss =  - dLdy * np.log(self.p+eps) - (1 - dLdy) * np.log(1 - self.p+eps)
        #print('Loss:', np.sum(np.sum(Loss)/dLdy.shape[0]))
        dLdX = self.p-dLdy
        return dLdX
        #### YOUR CODE HERE
        #pass

class MSE_Error:
    # Saves X for backprop, X.shape = N x 1
    def forward(self, X):
        self.X = X
        return X

    # Returns dL/dy (y - true labels)
    def backward(self, y):
        #Loss = np.sum((self.X-y)**2)
        #### YOUR CODE HERE
        dLdy= np.zeros((y.size, y.max()+1))
        for i in range(y.size):
            dLdy[i][y[i]]=1
        #Loss = np.sum((self.X-dLdy)**2)
        #print (Loss)
        dLdx = 2*self.X-2*dLdy
        return dLdx
        #pass
    def get_loss(self, y):
        dLdy= np.zeros((y.size, y.max()+1))
        for i in range(y.size):
            dLdy[i][y[i]]=1
        #print (dLdy)
        Loss =  (self.X-dLdy)**2
        #print('Loss:', np.sum(Loss))
        return np.sum(Loss)/dLdy.shape[0]


## Main class
# loss_function can be None - if the last layer is SoftMax_NLLLoss: it can produce dL/dy by itself
# Or, for example, loss_function can be MSE_Error()
class NeuralNetwork:
    def __init__(self, modules, loss_function=None):
        '''
        Constructs network with *modules* as its layers
        '''
        #### YOUR CODE HERE
        #pass
        self.layers = modules

    def forward(self, X):
        current_value = X
        for i in range(len(self.layers)):
            #print (self.layers[i])
            current_value = self.layers[i].forward(current_value)
        return current_value
        #### YOUR CODE HERE
        #### Apply layers to input
        #pass

    # y - true labels.
    # Calls backward() for each layer. dL/dy from k+1 layer should be passed to layer k
    # First dL/dy may be calculated directly in last layer (if loss_function=None) or by loss_function(y)
    def backward(self, y):
        #### YOUR CODE HERE
        # Преобразуем в единичную матрицу
        dLdy = y
        for i in range(len(self.layers)):
            # с конца в начало
            dLdy = self.layers[len(self.layers)-i-1].backward(dLdy)
        return dLdy

    # calls step() for each layer
    def step(self, learning_rate):
        for i in range(len(self.layers)-1):
            self.layers[i].step(learning_rate)
        pass
    def get_loss(self, y, numb):
        return  self.layers[len(self.layers)-1].get_loss(y, numb)