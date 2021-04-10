In many situations, the relationship between some covariate(s) X and y is not so straightforward. For linear models, the functions are simple a product between a constant beta and the value of the feature, and when you train the model, you are learning those coefficients, beta. However, attempting to fit a standard OLS regression on non-linear data may not capture the relationship very well. In these cases, there are a few approache we can take, such as variable selection, fitting rather a polynomial regression or applying different penalizing kernels, etc. In this page, we will explore the Generalized Additive Model and Nadaraya-Watson Kernel-Weighted regressions to see how the models perform on our real-life dataset. 


## Physicochemical Properties of Protein Tertiary Structure Data
The dataset we will be using is from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure).

|Column | Attribute| 
|--------|-------------|
|RMSD|Size of the residue             |
|F1 | Total surface area        |
|F2 | Non polar exposed area        |
|F3 | Fractional area of exposed non polar residue|
|F4 | Fractional area of exposed non polar part of residue|
|F5 | Molecular mass weighted exposed area|
|F6 | Average deviation from standard exposed area of residue|
|F7 | Euclidian distance|
|F8 | Secondary structure penalty|
|F9 | Spacial Distribution constraints (N,K Value)|



# Generalized Additive Model

GAMs are a class of statistical models in which the usual linear relationship between the features and the target is replaced by several non-linear smooth functions to model and capture the non-linearities in the data. Thus, GAMS can accomodate a non-linear relationship between input and output features by assigning a possibly non-linear function for each of the input features in the linear model. Unlike LOESS, GAMs use automatic smoothness selection methods to objectively determine the complexity of the fitted trend and allow for potentially complex, non-linear trends, a proper accounting of model uncertainty, and the identification of periods of significant temporal change.

The basic idea of splines is that we are going to fit smooth non-linear functions on a bunch of predictors Xi to capture and learn the non-linear relationships between the model’s variables X and Y. 'Additive' means we are going to fit and retain the additivity of the Linear Models.

The Regression Equation becomes:

![CodeCogsEqn (3)](https://user-images.githubusercontent.com/66886936/114239589-42166a80-9954-11eb-873c-e510cfb9513e.gif)


where *Y* is the dependent variable, *E(Y)* denotes the expected value, and *g(Y)* denotes the function that links the expected value to the predictor variables *x1,...,xp*, and the terms *s1(x1),...,sp(xp)* denote smooth, nonparametric functions.

We will need to install pygam and import the LinearGAM package, and we will examine the R-squared value as our metric for assessing the quality of our predictions. 

```python
!pip install pygam
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score as R2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from pygam import LinearGAM
```
First, we will take a look at a single split of the data and fit it into our GAM model. 

```python
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.1,random_state=1234)
gam = LinearGAM(n_splines=6).gridsearch(X_train, y_train,objective='GCV')
yhat_test = gam.predict(X_test)
print(R2(y_test, yhat_test))
```
```
100% (11 of 11) |########################| Elapsed Time: 0:00:10 Time:  0:00:10
0.33417590791145935
```
Here, we have a pretty weak model in terms of our coefficient of determination. Let us see if we can improve or verify our R-squared value with k-fold cross validation. In order to implement k-fold cross validation, we first create our own sklearn function. 

```python
from pygam import LinearGAM
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```
```
class GAMsk(BaseEstimator, RegressorMixin):
    def __init__(self, ns=None):
        self.ns = ns
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)

        if self.ns is None:
            gam = LinearGAM(n_splines=3).gridsearch(X, y,objective='GCV')
            self.pred_ = gam
        else:
            gam = LinearGAM(n_splines=self.ns).gridsearch(X, y,objective='GCV')
            self.pred_ = gam
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.pred_.predict(X).reshape(-1,1)
```
Now we can define a k-fold function using our sklearn GAM model:

```python
def DoKFold(X,y,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = GAMsk(ns=50)
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)
    PE.append(R2(y_test,yhat_test))
  return np.mean(PE)
```
```
DoKFold(X,y,10)
```
```
0.36723977496712423 
```
A little better!


To understand our results further, we will plot the residuals from a single train and test split using the yellowbrick library. Perhaps from the plot we can see that the distribution of our residuals is not quite normal. It is skewed right, which explains the low R-quared value. 

<img width="700" alt="download" src="https://user-images.githubusercontent.com/66886936/114251767-40f33680-9970-11eb-8648-ea466fe8b477.png">





# Nadaraya Watson Kernel Weighted Regression 

Kernel regression is a non-parametric alternative to least squares in which the predictor does not take a predetermined form. Kernel regression estimates the response by convolving the data with a kernel function to combine the influence of the individual data points. The Nadaraya-Watson regressor estimates the response y using a kernel weighted average of the individual datapoints *(xi,yi)*:

![CodeCogsEqn (4)](https://user-images.githubusercontent.com/66886936/114253738-646eaf00-9979-11eb-87b1-f8b95b7307db.gif)

where *K(x,x')* is a kernel function.

