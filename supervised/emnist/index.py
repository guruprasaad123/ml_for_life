import numpy as np
import matplotlib as plt
import pandas as pd

X_FNAME = "./data/alphanum-hasy-data-X.npy"
Y_FNAME = "./data/alphanum-hasy-data-y.npy"
SYMBOL_FNAME = "./data/symbols.csv"

X = np.load(X_FNAME)
Y = np.load(Y_FNAME)
SYMBOLS = pd.read_csv(SYMBOL_FNAME,usecols=[0,1])

print(np.unique(Y))
