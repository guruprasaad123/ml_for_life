## Stock Price Prediction

### Overview

This repo demonstrates a small example of time series prediction using [Keras api](https://keras.io/) with [tensorflow](https://www.tensorflow.org/) as backend

### Dependencies

- keras_2.2.4 ( [https://pypi.org/project/Keras/](https://pypi.org/project/Keras/) )

- pandas_0.24.1 ( [https://pypi.org/project/pandas/](https://pypi.org/project/pandas/) )

- numpy_1.16.2 ( [https://pypi.org/project/numpy/](https://pypi.org/project/numpy/) )

- tensorflow_1.13.1 ( [https://pypi.org/project/tensorflow-gpu/](https://pypi.org/project/tensorflow-gpu/) )

- scikit-learn_0.20 ( [https://pypi.org/project/scikit-learn/](https://pypi.org/project/scikit-learn/) )

- matplotlib_3.0.3( [https://pypi.org/project/matplotlib/](https://pypi.org/project/matplotlib/) )

### Dataset

The Real-time stocks prices of Carriage Services, Inc. from  [ 3/20/2014 to 3/25/2019 ] (https://in.finance.yahoo.com )

### Installation

Install all the dependencies using this command

```python
pip install -t lib -r requirements.txt
```

or

```python
pip install -r requirements.txt
```

### Running the App

Run the app locally using this command

```python
python index.py
```

if you have [jupyter notebook](https://jupyter.org/) installed on your machine , you can opening the `predict.ipynb` file

Resources :
http://sebastianruder.com/optimizing-gradient-descent/
https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53
https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
