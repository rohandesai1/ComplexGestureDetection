import numpy as np
from math import log10

def logOrLinRegression(x, y, initial_w, initial_b, iterations, alpha, logistic, normalize):


    def compute_cost(x, y, w, b):
        m = x.shape[0]
        
        cost = 0
        for i in range(m):
            prediction = np.dot(x[i], w) + b
            actual = y[i]
            if logistic:
                error = (actual * log10(prediction) + ((1 - actual) * log10(1 - prediction))) 
            else:
                error = abs(prediction - actual) 
            cost += error
        if logistic:
            cost = -cost / (2 * m)
        cost = cost / (2 * m)

        return cost


    def z_score_normalization(x):
        
        mean = np.mean(x, axis = 0)
        std = np.std(x, axis = 0)
        
        x_normalized = (x - mean)/std
        
        return x_normalized, mean, std
    
    def sigmoid(z):
        return 1/ (1 + np.exp(-z))

    def compute_gradient(x, y, w, b):
        twoDimensional = True
        try:
            m,n = x.shape

        except ValueError:
            m = x.shape[0]
            n = 1
            twoDimensional = False
            

        w_gradient = np.zeros(n) # = 0  
        b_gradient = 0


        for i in range(m):
            if logistic:
                error =  (sigmoid(np.dot(x[i], w) + b) - y[i])

            else:
                error = (np.dot(x[i], w) + b) - y[i] 
  
            
            for j in range(n):
                if twoDimensional:

                    w_gradient[j] = w_gradient[j] + error * x[i, j]
                else:
                    w_gradient[j] = w_gradient[j] + error * x[i]
            b_gradient += error 
            
        w_gradient = w_gradient/m
        b_gradient = b_gradient/m
        
        return w_gradient, b_gradient
    def copy(target):
        return target
        


   
    w = copy(initial_w)
    b = copy(initial_b)
    cost_history = []
    if normalize:
        x, mean, std = z_score_normalization(x)


    for i in range(iterations):

        w_change, b_change = compute_gradient(x, y, w, b)
        
        #cost = compute_cost(x, y, w, b)
        
        w = w - alpha * w_change
        b = b - alpha * b_change
        
      #  cost_history.append(cost)
                

    
    if normalize:
        return w, b, cost_history, mean, std
    else:
        return w,b,cost_history


def predict(factors, w, b, mean, std, logistic):
    if logistic:
        prediction = sigmoid(np.dot(w, (factors - mean)/std) + b)
    else:
        prediction = np.dot(w, (factors - mean)/std) + b # subtract mean and divide by std to remove normalized values (model was trained on normalized values)
    return prediction



def sigmoid(z):
    return (1/ (1 + np.exp(-z)))
 