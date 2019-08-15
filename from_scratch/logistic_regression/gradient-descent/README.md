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

!
loss= \frac{1}{m}(y-y_{pred})^{2}


For Logistic Regression ,

$$
loss = \frac{1}{m}(-ylog(h_{\theta}(x) \ -(1-y)( \ log(1-h_{\theta}(x) \ ) \ ) \ )
$$




