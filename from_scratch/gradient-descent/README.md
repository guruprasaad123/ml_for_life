# Gradient Descend

This project is all about learning how gradient works.So the best way to learn something is to implement it from scratch so in this project we'll implement `batch-gradient-descent`algorithm which learns mapping between the two features and provides the line-of-best-fit (linear-regression).The Dataset represents `cycled distance vs calorie burned`.

## Dependencies
- [numpy == 1.16.2](https://pypi.org/project/numpy/) 
- [matplotlib == 2.2.2](https://pypi.org/project/matplotlib/)

you can also install these requirements using `requirements.txt` using this command below.

```
pip install -r requirements.txt
```

#### Gradient descent visualization
![gd](https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif)

#### Sum of squared distances formula (to calculate our error)
![error](https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png)

#### Partial derivative with respect to b and m (to perform gradient descent)
![error2](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)
