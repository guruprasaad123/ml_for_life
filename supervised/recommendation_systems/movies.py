import pandas as pd
import numpy as np
from  scipy.sparse import coo_matrix 

def fetch_movies():
    ratings = pd.read_csv('ratings.dat',sep='::')

    n_ratings  =  ratings.shape[0]

    data , row , col = [],[],[]
    users = {}
    for n in range(n_ratings):
        if( ratings['Rating'][n] >= 4 ):
            user = ratings['UserID'][n]
            movie = ratings['MovieID'][n]
            if user in users:
                users[user] = users[user]+1
            else :
                users[user]=1
            
            data.append(users[user])
            row.append(user)
            col.append(movie)

    coo = coo_matrix( (data,(row,col)))
    return coo