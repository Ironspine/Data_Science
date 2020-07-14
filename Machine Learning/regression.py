import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearReg:
    
    '''
    It uses a simple ordinary least squares method when only one independent predictor is implemented. 
    '''
    
    def fit(self, X_train, y_train):
        
        '''
            Fits a linear regression using 1 predictor.
        '''
        self.X_train = X_train
        self.y_train = y_train
        
        if len(X_train) != len(y_train):
            raise ValueError('Lenght of numpy arrays must be same!')
            
        X_squares = sum([i**2 for i in X_train])
        Xy_squares = sum([i*j for i,j in zip(X_train, y_train)])
        
        self.gradient = float(len(X_train) * Xy_squares - sum(X_train) * sum(y_train)) / (len(X_train) * X_squares - sum(X_train)**2)

        self.intersection = float((sum(self.y_train) - self.gradient * sum(X_train)) / len(X_train))

        print("Equation of the fitted linear function is y = %.4f * x + %.4f" %(self.gradient, self.intersection))

    def plot(self, x_axis_name, y_axis_name):

        '''
            Plots the fitted linear regression.
        '''

        X = np.arange(min(self.X_train), max(self.X_train), 0.01)
        y = self.gradient * X + self.intersection

        fig = plt.figure(figsize = (8, 6))
        plt.plot(X, y, 'green', linewidth = 1.5)
        plt.plot(self.X_train, self.y_train, 'o',color = 'blue')
        plt.grid(b = None, which = 'major', axis = 'both')
        plt.title("Least square method", size = 18)
        plt.xlabel(str(x_axis_name), size = 12)
        plt.ylabel(str(y_axis_name), size = 12)
        plt.show()

    def predict(self, X_test):
        
        '''
            Accepts a 1-dimensonal numpy array.
        '''
        
        y_pred = np.array([])
        for i in X_test:
            y_pred = np.append(y_pred, self.gradient * i + self.intersection)
        return y_pred
    
    def corr(self, X, y):

        '''
            Computes Pearson-correlation between the two data sets.
        '''

        print('Pearson-correlation: %.4f' %round(np.corrcoef(X, y)[0][1], 4))

    def residuals(self):

        '''
            Computes a list of residuals and their sum.
        '''

        residuals = np.array([])
        for i, j in zip(self.X_train, self.y_train):
            residual = abs((self.gradient * i + self.intersection) - j)
            residuals = np.append(residuals, round(residual, 3))
        print('List of residuals: {0} \nSum of residuals: {1}' .format(residuals, round(sum(residuals), 3)))

    def mae(self, y_true, y_pred):
        summ = sum([abs(i - j) for i, j in zip(y_true, y_pred)])
        mae = summ / len(y_true)
        print('Mean absolut error: %.4f' %mae)

    def mse(self, y_true, y_pred):
        summ = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        mse = summ / len(y_true)
        print('Mean squared error: %.4f' %mse)

    def rmse(self, y_true, y_pred):
        summ = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        rmse = (summ / len(y_true))**0.5
        print('Root mean squared error: %.4f' %rmse)
        
    def r2_score(self, y_true, y_pred):
        ss_total = sum([(i - np.mean(y_true))**2 for i in y_true])
        ss_residual = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        print( 'R_squared score: ', 1 - ss_residual/ss_total)
    
class MultipleLinearReg:
    
    '''
    It uses multiple linear regression with ordinary least squares method when multiple independent predictors are used. 
    '''
    
    def fit(self, X_train, y_train):
        
        '''
            Accepts 2-dimensonal numpy array as a predictor and 1-dimensonal numpy array as the dependent variable.
        '''
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('Sizes do not match: size of X_train is ' + str(X_train.shape[0]) + ' which is different from size of y_train ' + str(y_train.shape[0]))
                             
        coeff_vector = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train)
        self.coeff_vector = coeff_vector
        print('Coefficients: ', self.coeff_vector)
    
    def predict(self, X_test):
        
        '''
            Accepts a 2-dimensonal numpy array.
        '''
        
        y_pred = np.matmul(X_test, self.coeff_vector)
        return y_pred
              
    def mae(self, y_true, y_pred):
        
        summ = sum([abs(i - j) for i, j in zip(y_true, y_pred)])
        mae = summ / len(y_true)
        print('Mean absolut error: %.4f' %mae)

    def mse(self, y_true, y_pred):
        summ = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        mse = summ / len(y_true)
        print('Mean squared error: %.4f' %mse)

    def rmse(self, y_true, y_pred):
        summ = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        rmse = (summ / len(y_true))**0.5
        print('Root mean squared error: %.4f' %rmse)
        
    def r2_score(self, y_true, y_pred):
        ss_total = sum([(i - np.mean(y_true))**2 for i in y_true])
        ss_residual = sum([(i - j)**2 for i, j in zip(y_true, y_pred)])
        print('R_squared score: ', 1 - ss_residual/ss_total)