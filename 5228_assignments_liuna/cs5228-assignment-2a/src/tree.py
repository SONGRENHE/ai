import numpy as np

        
        
class MyDecisionStumpRegressor:
    
    def __init__(self, num_thresholds=None):
        self.num_thresholds = num_thresholds
        self.feature_idx = None
        self.threshold = None
        self.y_left, self.y_right = None, None
        
        
    def calc_rss_score(self, y_left, y_right):
        """
        Calculate the RSS score (impurity) of a list of values
        (RSS = residual sum of squares)

        Inputs:
        - y: A numpy array of shape (N,) containing the sample values, 
             where N is the number of samples in this node
             
        Returns:
        - Scalar value representing the RSS score of a list of values. 
        """        

        # Just a failsafe: if 1 child node is empty => invalid split => return infinite RSS
        if len(y_left) == 0 or len(y_right) == 0:
            return np.inf
        
        rss = None

        #########################################################################################
        ### Your code starts here ###############################################################

        y_left_mean = np.mean(y_left)
        y_right_mean = np.mean(y_right)
        rss = np.sum((y_left-y_left_mean)**2) + np.sum((y_right-y_right_mean)**2)

        
        ### Your code ends here #################################################################
        #########################################################################################

        return rss

    
        
    def calc_thresholds(self, values):
        """
        Identifies the threshold for splitting given an array of feature values
        (RSS = residual sum of squares)

        Inputs:
        - values: A numpy array of shape (N,) containing the feature values, 
                  where N is the number of samples
             
        Returns:
        - thresholds: A set of size M where M=self.threshold
                      or M=N if self.threshold is None
        """   

        k = self.num_thresholds
        
        thresholds = set()
        
        #########################################################################################
        ### Your code starts here ###############################################################
        # print(values)
        # print(np.sort(values))
        if k is None:
            thresholds = set(np.unique(values))
        else:
            split_list = np.array_split(np.sort(np.unique(values)), k+1)
            first = list()
            last = list()
            for i in range(len(split_list)):
                if i != 0:
                    first.append(split_list[i][0])
                if i != len(split_list) - 1:
                    last.append(split_list[i][-1])
            
            thresholds = set(np.add(last, first)/2)

        
        ### Your code ends here #################################################################
        #########################################################################################
        
        return thresholds
        
        
        
        
    def fit(self, X, y):
        """
        Trains the Decision Tree Regressor

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
        
        Returns:
        - self (but calculates self.feature_idx, self.threshold, self.y_left, self.y_right!!!)
        """     
        
        best_rss = np.inf
        
        ## Loop through all features (columns of X) to find the best split
        for feature_idx in range(X.shape[1]):
            
            # Get all values for current features
            values = X[:, feature_idx]
            
            #####################################################################################
            ### Your code starts here ###########################################################                         
            possible_thresholds = self.calc_thresholds(values)
            rss = list()
            for i in possible_thresholds:
                y_left = y[values<=i]
                y_right = y[values>i]
                rss.append(self.calc_rss_score(y_left, y_right))
            
            if min(rss) < best_rss:
                best_rss = min(rss)
                self.feature_idx = feature_idx
                self.threshold = list(possible_thresholds)[np.argmin(rss)]
                self.y_left = y[values<=self.threshold]
                self.y_right = y[values>self.threshold]

            
            ### Your code ends here #############################################################
            #####################################################################################
        ## Return MyDecisionStumpRegressor object
        return self            
            
                    
    def predict(self, X):
        """
        Predict labels for a set of samples

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
             
        Returns:
        - y_pred: A numpy array (N,) containing the N predicted class labels (numerical labels!)
        """            
        
        # We initalize y_pred the NaN values to better check the results
        y_pred = np.full((X.shape[0], ), np.nan, dtype=np.float64)
        
        ## Loop over all sample in X and predict the class
        for idx, x in enumerate(X):
            
            #####################################################################################
            ### Your code starts here ###########################################################       
            if x[self.feature_idx] <= self.threshold:
                y_pred[idx] = np.mean(self.y_left)
            else:
                y_pred[idx] = np.mean(self.y_right)

            
            ### Your code ends here #############################################################
            #####################################################################################
            
            pass # Just here so the empty loop does not throw an error
                    
        return y_pred              
    
    
    
    
    
        
################################################################################################
################################################################################################
###
### GradientBoostedRegressor
###
################################################################################################
################################################################################################
    
    
class MyGradientBoostedRegressor:
    
    def __init__(self, learning_rate=0.1, n_estimators=100):
        self.stumps = []
        self.lr = learning_rate
        self.n_estimators = n_estimators
        self.initial_f = None
        
    
    def fit(self, X, y):
        
        """
        Trains the Gradient-Boosted Regrossor using MyDecisionStumpRegressor

        Inputs:
        - X: A numpy array of shape (N, F) containing N numerical values for F features
             representing the feature values
        - y: A numpy array of shape (N,) containing N labels
        
        Returns:
        - self (but calculates and keeps track of all self.n_estimators decision stumps in self.trees!!!)
        """     
        
        self.stumps = []
        
        self.initial_f = y.mean()
        # Set initial prediction f_0(x_i) to mean for all data points
        f = np.array([ self.initial_f ]*X.shape[0])

        for m in range(1, self.n_estimators+1):

            #####################################################################################
            ### Your code starts here ###########################################################             
            
            ## Use your implementation of MyDecisionStumpRegressor in here!
            stump = MyDecisionStumpRegressor().fit(X, y-f)

            h = stump.predict(X)
            f = f + self.lr*h
            self.stumps.append(stump)
            
            
            ### Your code ends here #############################################################
            #####################################################################################
            
            pass # Just here so the empty loop does not throw an error
       
        return self
    
    
    def predict(self, X):
        
        y_pred = np.array([self.initial_f]*X.shape[0])
        
        #########################################################################################
        ### Your code starts here ###############################################################
        for i in range(len(self.stumps)):
            y_pred = y_pred + self.stumps[i].predict(X)*self.lr
            
        ### Your code ends here #################################################################
        #########################################################################################        
        
        return y_pred
    
