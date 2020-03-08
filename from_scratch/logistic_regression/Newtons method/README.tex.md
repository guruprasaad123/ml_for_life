# Logistic Regression

In this repo i have performed *Logistic regression* using *gradient-descent* which is optimization method on a sample dataset that we generated on our own.

## Dependencies

- numpy

- pandas

- matplotlib

We use numpy for mathematical computation , pandas for framing the sample dataset that we generated . matplotlib to visualize the loss which reduces over training.

## Under the hood

 *loss function* for logistic regression is different from linear regression.

For Linear Regression ,

$$
loss= \frac{1}{m}(y-y_{pred})^{2}
$$

For Logistic Regression ,

$$
Cost / Loss =
 \begin{cases}
 -log(h_{\theta}(x)) & \text{if $y$ is 1} \\
 -log(1-h_{\theta}(x)) & \text{if $y$ is 0}
 \end{cases}
$$

Which can be simplified as below ,

$$
loss = \frac{1}{m}(-ylog(h_{\theta}(x) \ -(1-y)( \ log(1-h_{\theta}(x) \ ) \ ) \ ) 
$$


 let's check this figure below and see how the Cost / Loss function works .

 when $y = 1$ 

![mtVFHjoeQkaNuKP](https://i.loli.net/2019/08/27/mtVFHjoeQkaNuKP.png)

if $h(x)$ is close to $1$ , cost is so low wherelse when $h(x)$ is close to 0 , cost is so high 

When $y=0$ ,

![EumjY89IzUcVh1S](https://i.loli.net/2019/08/27/EumjY89IzUcVh1S.png)

if $h(x)$ is close to $0$ , cost is so low wherelse when $h(x)$ is close to $1$ , cost is so high , make sense now !

## Newton's Method

Newtons method is a *second order optimisation method* which is quite often used for

- Root finding problems

- Optimisation problems

We used Newtons method here for optimisation in place for gradient Descent which is a *first order optimisation method*

Gradient Descent we use the *first order deriavative* to optimize the weights where in the Newtons method we use  

*second order deriavative* to optimize the weights.

$$
W_{new} = W_{old} - ( \frac{\partial ^2 L}{\partial W^2} ) ^{-1} ( \frac{\partial L}{\partial W} )
$$

References :

[https://math.stackexchange.com/questions/3314437/hessian-of-loss-function-applying-newtons-method-in-logistic-regression/3321299#3321299](https://math.stackexchange.com/questions/3314437/hessian-of-loss-function-applying-newtons-method-in-logistic-regression/3321299#3321299)

[http://mathgotchas.blogspot.com/search/label/Logistic%20Regression](http://mathgotchas.blogspot.com/search/label/Logistic%20Regression)

[https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function](https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function)

[https://math.stackexchange.com/questions/2318003/hessian-of-the-logistic-regression-cost-function](https://math.stackexchange.com/questions/2318003/hessian-of-the-logistic-regression-cost-function)
