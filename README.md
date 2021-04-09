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

First, we will take a look at a single split of the data and fit it into our GAM model. We will need to install pygam and import the LinearGAM package.  

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
```python
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.1,random_state=1234)
gam = LinearGAM(n_splines=6).gridsearch(X_train, y_train,objective='GCV')
yhat_test = gam.predict(X_test)
print(R2(y_test, yhat_test))
```



