# Building Recommendation Systems

While building Recommendation systems  , there are 2 types of recommadation systems 

- Collaborative ( Based on other liked in past )

- content-based ( Based on the content you like )

- Collaborative + content-based ( hybrid  = mixture of both )

Most of the web-services does that these days.

We'll be building recommendation systems based on a real-time-dataset.we have downloaded the datasets from this [site](https://grouplens.org/datasets/movielens/).And look at [warp_loss](_https://lyst.github.io/lightfm/docs/examples/warp_loss.html).

Four loss functions are available:

- logistic: useful when both positive (1) and negative (-1) interactions are present.

- BPR: Bayesian Personalised Ranking [1]_ pairwise loss. Maximises the prediction difference between a positive example and a randomly chosen negative example. Useful when only positive interactions are present and optimising ROC AUC is desired.

- WARP: Weighted Approximate-Rank Pairwise [2]_ loss. Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found. Useful when only positive interactions are present and optimising the top of the recommendation list (precision@k) is desired.

- k-OS WARP: k-th order statistic loss [3]_. A modification of WARP that uses the k-th positive example for any given user as a basis for pairwise updates.

Our task is to build recommendation systems i.e `Collaborative and Content-based` using [lightfm](http://lyst.github.io/lightfm/docs/home.html), which is allowed by warp loss functions .

## Installation

Install the lib that are neccessary by using this cmd.

```python
pip install -r requirements.txt -t lib
```

or

```python
pip install -r requirements.txt
```

## Running the app

`recommender.py` is the main file so all you need to do is , issue this command

```python
python recommender.py
```

which like outputs this.

![5c93bcb60ac86](https://i.loli.net/2019/03/22/5c93bcb60ac86.png)

or ,

if you have installed [jupyter notebook](https://jupyter.org/) installed on your machine , try running  `recommender.ipynb`

Resources :

https://github.com/llSourcell/recommender_system_challenge
