In many situations, the relationship between some covariate(s) X and y is not so straightforward. For linear models, the functions are simple a product between a constant beta and the value of the feature, and when you train the model, you are learning those coefficients, beta. However, attempting to fit a standard OLS regression on non-linear data may not capture the relationship very well. In these cases, there are a few approache we can take, such as variable selection, fitting rather a polynomial regression or applying different penalizing kernels, etc. In this page, we will explore the Generalized Additive Model and Nadaraya-Watson Kernel-Weighted regressions to see how the models perform on our real-life dataset. 


# Generalized Additive Model

GAMs are a class of statistical models in which the usual linear relationship between the features and the target is replaced by several non-linear smooth functions to model and capture the non-linearities in the data. Thus, GAMS can accomodate a non-linear relationship between input and output features by assigning a possibly non-linear function for each of the input features in the linear model. Unlike LOESS, GAMs use automatic smoothness selection methods to objectively determine the complexity of the fitted trend and allow for potentially complex, non-linear trends, a proper accounting of model uncertainty, and the identification of periods of significant temporal change.




