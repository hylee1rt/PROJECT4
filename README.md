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

We will need to install pygam and import the LinearGAM package, and we will examine the R-squared value and the RMSE as our metric for assessing the quality of our predictions. 

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


To understand our results further, we will plot the residuals from a single train and test split using the yellowbrick library. Perhaps from the plot we can see that the distribution of our residuals is not quite normal. It is skewed right, which explains the low R-quared value. Next, we will see how our kernel density estimator performs.

<img width="700" alt="download" src="https://user-images.githubusercontent.com/66886936/114251767-40f33680-9970-11eb-8648-ea466fe8b477.png">




# Nadaraya Watson Kernel Weighted Regression 

Kernel regression is a non-parametric alternative to least squares in which the predictor does not take a predetermined form. Kernel regression estimates the response by convolving the data with a kernel function to combine the influence of the individual data points. The Nadaraya-Watson regressor estimates the response y using a kernel weighted average of the individual datapoints *(xi,yi)*:

![CodeCogsEqn (5)](https://user-images.githubusercontent.com/66886936/114253879-4a819c00-997a-11eb-88e9-d00be2ab5f26.gif)


where *Kh* is a kernel function with a bandwith *h* and the denominator is a weighting term with sum 1. Common derivation is as follows:

![CodeCogsEqn (7)](https://user-images.githubusercontent.com/66886936/114253949-b6640480-997a-11eb-9578-86ef823e6acf.gif)

![CodeCogsEqn (8)](https://user-images.githubusercontent.com/66886936/114254007-017e1780-997b-11eb-8779-e1f8410a00cb.gif)

![CodeCogsEqn (9)](https://user-images.githubusercontent.com/66886936/114254039-22466d00-997b-11eb-9553-e3f469400d78.gif)

![CodeCogsEqn (11)](https://user-images.githubusercontent.com/66886936/114254078-5883ec80-997b-11eb-953c-e5f53689e980.gif)


Because the implementation of this Nadaraya-Watson kernel density estimator lacks in optimization, the function requires a LOT of RAM space and causes the runtime to crash. Due to this limitation, we will have to determine the optimal gamma value first using GridSearchCV. 

```python 
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=1234)
Xs_train = scale.fit_transform(X_train)
Xs_test = scale.transform(X_test)
ys_train = scale.fit_transform(y_train)
ys_test = scale.transform(y_test)

param_grid=dict(kernel=["rbf"],gamma=np.linspace(30, 50, 10))
model = GridSearchCV(NadarayaWatson(), cv=5, param_grid=param_grid)
model.fit(Xs_train,ys_train)
model.best_params_
```
{'gamma': 32.22222222222222, 'kernel': 'rbf'}

Using these parameters we will define a k-fold function:

```python
def DoKFold(X,y,k):
  scale = StandardScaler()
  X = scale.fit_transform(X)
  y = scale.fit_transform(y)
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    model = NadarayaWatson(kernel='rbf', gamma=32.2222)
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)
    PE.append(R2(y_test,yhat_test))
  return np.mean(PE)
  ```
  ```
  DoKFold(X,y,10)
  ```
  0.6291443207235665

Let's now look at the residuals for a single train and test split of the data. Clearly, the R-quared value for the training set is much higher and therefore our model is highly overfit. However, our testing R-squared value performed much better than the GAM model. We can also note that the distribution of the residuals looks like a normal distribution, which may explain our higher R-squared result. 

<img width="700" alt="download" src="https://user-images.githubusercontent.com/66886936/114254669-c41b8900-997e-11eb-98c5-f0a4620b324d.png">




# Results

| Model                          | RMSE      |  R-squared      |
|--------------------------------|-----------|--------------------|
| GAM                          |      4.9367 | 0.3672           |                     
| Nadaraya-Watson               |      0.03 | 0.6292          |    
 


