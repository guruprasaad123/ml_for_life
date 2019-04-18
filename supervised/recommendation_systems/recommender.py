import numpy as np
import pandas as pd
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from movies import fetch_movies
import random 
movies = fetch_movies()

model = LightFM(loss="warp")
model.fit(movies,epochs=30,num_threads=2)
 

users = pd.read_csv('users.dat',sep='::')
movies_data = pd.read_csv('movies.dat',sep='::')

user1 = random.choice(users['UserID'])
user2 = random.choice(users['UserID'])
user3 = random.choice(users['UserID'])

def get_recommendation(users,model,movies_matrix,movies_data):
    n_items = movies_matrix.shape[1]
    for user in users:
        scores = model.predict(user,np.arange(n_items))
        topscore = np.argsort(-scores)[:3]

        print('For User ',user)
        print('\t Reccomanded Movies :')

        for movie in topscore:
            movie_index = np.where(movie == movies_data['MovieID'])[0]
            movie_title = movies_data['Title'][movie_index[0]]
            print('\t ',movie_title)




get_recommendation([user1,user2,user3],model,movies,movies_data)