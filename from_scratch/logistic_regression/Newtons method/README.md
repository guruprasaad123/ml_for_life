# Logistic Regression

In this repo i have performed*Logistic regression* using *gradient-descent* which is optimization method on a sample dataset that we generated on our own.

## Dependencies

- numpy

- pandas

- matplotlib

We use numpy for mathematical computation , pandas for framing the sample dataset that we generated . matplotlib to visualize the loss which reduces over training.

## Under the hood

 *loss function* for logistic regression is different from linear regression.

For Linear Regression ,

<p align="center"><img src="/from_scratch/logistic_regression/gradient-descent/tex/f81a893ca8e5ebd4cae3c4cc919d31ee.svg?invert_in_darkmode&sanitize=true" align=middle width=152.1768435pt height=32.990165999999995pt/></p>

For Logistic Regression ,

<p align="center"><img src="/from_scratch/logistic_regression/gradient-descent/tex/1b57fa57f751650d4abf4efd3691b701.svg?invert_in_darkmode&sanitize=true" align=middle width=297.4605216pt height=49.315569599999996pt/></p>

Which can be simplified as below ,

<p align="center"><img src="/from_scratch/logistic_regression/gradient-descent/tex/707cdd965302e7de9a5a2b557de7be0c.svg?invert_in_darkmode&sanitize=true" align=middle width=382.0617009pt height=32.990165999999995pt/></p>

<p align="center"><img src="/from_scratch/logistic_regression/gradient-descent/tex/e7e1fce898b1583cb28cc71db94ffdd5.svg?invert_in_darkmode&sanitize=true" align=middle width=0.0pt height=0.0pt/></p>

 let's check this figure below and see how the Cost / Loss function works .

 when <img src="/from_scratch/logistic_regression/gradient-descent/tex/28bfa1de0b829a8ef9aebb8eb6eb92a3.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/> 

![mtVFHjoeQkaNuKP](https://i.loli.net/2019/08/27/mtVFHjoeQkaNuKP.png)

if <img src="/from_scratch/logistic_regression/gradient-descent/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> is close to <img src="/from_scratch/logistic_regression/gradient-descent/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> , cost is so low wherelse when <img src="/from_scratch/logistic_regression/gradient-descent/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> is close to 0 , cost is so high 

When <img src="/from_scratch/logistic_regression/gradient-descent/tex/a42b1c71ca6ab3bfc0e416ac9b587993.svg?invert_in_darkmode&sanitize=true" align=middle width=38.78604674999999pt height=21.18721440000001pt/> ,

![EumjY89IzUcVh1S](https://i.loli.net/2019/08/27/EumjY89IzUcVh1S.png)

if <img src="/from_scratch/logistic_regression/gradient-descent/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> is close to <img src="/from_scratch/logistic_regression/gradient-descent/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> , cost is so low wherelse when <img src="/from_scratch/logistic_regression/gradient-descent/tex/82b61730744eb40135709391ec01cbdb.svg?invert_in_darkmode&sanitize=true" align=middle width=31.651535849999988pt height=24.65753399999998pt/> is close to <img src="/from_scratch/logistic_regression/gradient-descent/tex/034d0a6be0424bffe9a6e7ac9236c0f5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> , cost is so high , make sense now !

## Newton's Method

Newtons method is a *second order optimisation method* which is quite often used for

- Root finding problems

- Optimisation problems

We used Newtons method here for optimisation in place for gradient Descent which is a *first order optimisation method*

Gradient Descent we use the *first order deriavative* to optimize the weights where in the Newtons method we use  

*second order deriavative* to optimize the weights.

<p align="center"><img src="/from_scratch/logistic_regression/gradient-descent/tex/f090d1170d83613177d28a1cd32be2a9.svg?invert_in_darkmode&sanitize=true" align=middle width=229.82201385pt height=35.77743345pt/></p>

References :

[https://math.stackexchange.com/questions/3314437/hessian-of-loss-function-applying-newtons-method-in-logistic-regression/3321299#3321299](https://math.stackexchange.com/questions/3314437/hessian-of-loss-function-applying-newtons-method-in-logistic-regression/3321299#3321299)

[http://mathgotchas.blogspot.com/search/label/Logistic%20Regression](http://mathgotchas.blogspot.com/search/label/Logistic%20Regression)

[https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function](https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function)

[https://math.stackexchange.com/questions/2318003/hessian-of-the-logistic-regression-cost-function](https://math.stackexchange.com/questions/2318003/hessian-of-the-logistic-regression-cost-function)
