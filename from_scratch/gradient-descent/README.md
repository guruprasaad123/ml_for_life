# Gradient Descend

This project is all about learning how gradient works.So the best way to learn something is to implement it from scratch so in this project we'll implement `gradient-descent`algorithm which learns mapping between the two features and provides the line-of-best-fit (linear-regression).The Dataset represents `cycled distance vs calorie burned`.

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

#### Partial derivative with respect to weights m and bias b (to perform gradient descent)
![error2](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)

## Usage
Run this below code to get-started.

```
python3 demo.py
``` 
The hyperparameters that are involved in the process.
epochs = 1000 <br>
initial_m = 0 <br>
initial_b = 0 <br>
learning_rate = 0.001 <br>

   ```
Running
Error @ inital stage : 5565.107834483212
Running 0/1000
Running 100/1000
Running 200/1000
Running 300/1000
Running 400/1000
Running 500/1000
Running 600/1000
Running 700/1000
Running 800/1000
Running 900/1000
Error after Performing Gradient Descent : 112.61481011613475
   ```
#### For better visualization try this [file](https://github.com/guruprasaad123/ml_for_life/blob/master/from_scratch/gradient-descent/batch-gradient-descent.ipynb).<br>
if you have jupyter-notebook installed in your system , play with this [file](https://github.com/guruprasaad123/ml_for_life/blob/master/from_scratch/gradient-descent/batch-gradient-descent.ipynb)<br>

## Results

We have successfully created ``Line of best fit`` using gradient-descent.

![line of best fit](https://github.com/guruprasaad123/ml_for_life/blob/master/from_scratch/gradient-descent/Figure_1.png)

And also plotted the `valley`` and found out ``minima``.
![valley](https://github.com/guruprasaad123/ml_for_life/blob/master/from_scratch/gradient-descent/Figure_2.png)
