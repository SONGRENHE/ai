import numpy as np

from sklearn.metrics import mean_squared_error, log_loss


class MyLinearRegression:
    
    
    def __init__(self):
        self.theta = None
    
    
    def add_bias(self, X):
        
        ones = np.ones(X.shape[0]).reshape(-1, 1)

        return np.hstack([ones, X])

    
    
    def fit(self, X, y):
        """
        Computes the Normal Equation to find the best theta values analytically

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
        - y: A numpy array of shape (N,) containing N ground truth values
             
        Returns:
        - nothing (bet sets self.theta which should be a numpy array of shape (F+1,)
          containing the F+1 coefficients for the F features and the constant/bias term)
        """         

        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        theta = None

        #########################################################################################
        ### Your code starts here ###############################################################
        X_star = np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose())
        theta = np.dot(X_star,y)

        
        ### Your code ends here #################################################################
        #########################################################################################

        self.theta = theta
    
        return self

    
    
    
    
    
    
class MyLogisticRegression:
    
    
    def __init__(self):
        self.theta = None
    

    def add_bias(self, X):

        ones = np.ones(X.shape[0]).reshape(-1, 1)

        return np.hstack([ones, X])

    
    def calc_loss(self, y, y_pred):
        
        loss = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        loss = 1/y.shape[0]*np.sum(-y*np.log(y_pred)-(1-y)*np.log(1-y_pred))

        ### Your code ends here #################################################################
        #########################################################################################
        
        return loss

    
    def calc_h(self, X):
        
        h = None
        
        #########################################################################################
        ### Your code starts here ###############################################################        
        h = 1/(1 + np.exp(-np.dot(X, self.theta)))

    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return h
        

    def calc_gradient(self, X, y, h):
        
        grad = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        grad = (1/y.shape[0])*np.dot(X.transpose(),h-y)
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return grad

    
    
    
    def fit(self, X, y, lr=0.001, num_iter=100, verbose=False):
        """
        Fits a Logistic Regression model on a given dataset

        Inputs:
        - X: A numpy array of shape (N, F) containing N data samples with F features
        - y: A numpy array of shape (N,) containing N ground truth values
        - lr: A real value representing the learning rate
        - num_iter: A integer value representing the number of iterations 
        - verbose: A Boolean value to turn on/off debug output
             
        Returns:
        - nothing (bet sets self.theta which should be a numpy array of shape (F+1,)
          containing the F+1 coefficients for the F features and the constant/bias term)
        """      

        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1]).reshape(-1,1)

        for i in range(num_iter):

            #########################################################################################
            ### Your code starts here ###############################################################      
            
            h = self.calc_h(X)

            grad = self.calc_gradient(X, y, h)

            self.theta = self.theta - lr*grad


            ### Your code ends here #################################################################
            #########################################################################################        
            
            # Print loss every 10% of the iterations
            if verbose == True:
                if(i % (num_iter/10) == 0):
                    print('Loss: {:.3f} \t {:.0f}%'.format(self.calc_loss(y, h), (i / (num_iter/100))))

        # Print final loss
        print('Loss: {:.3f} \t 100%'.format(self.calc_loss(y, h)))
    
        return self
    
    
    def predict(self, X, threshold=0.5):
        
        # Add bias term x_0=1 to data
        X = self.add_bias(X)
        
        y_pred = None

        #########################################################################################
        ### Your code starts here ###############################################################
        
        h = self.calc_h(X)
        h[h>0.5] = 1
        h[h<=0.5] = 0
        y_pred = h
    
        ### Your code ends here #################################################################
        #########################################################################################
        
        return y_pred

    