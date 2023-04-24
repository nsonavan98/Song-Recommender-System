import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm
import pickle


song_db=pd.read_csv("/home/aditya/Desktop/rating1.csv")


song_db_subset=song_db.head(10000).copy()


song_db_subset['rating']=song_db_subset['rating'].apply(pd.to_numeric, errors='coerce')



train_data,test_data=train_test_split(song_db_subset,test_size=.20,random_state=0)

users=song_db_subset['user_id'].unique()
songs=song_db_subset['song'].unique()

songs_to_ix = { s:i for i,s in enumerate(songs) }
ix_to_songs = { i:s for i,s in enumerate(songs) }

print("no of users   ",len(users))
print("no of songs   ",len(songs))
#print(song_db_subset['rating'])
print("no of users in training set  ",len(train_data['user_id'].unique()))
print("no of songs in training set  ",len(train_data['song'].unique()))



temp=0;
user_matrix=[[0 for i in range(len(songs))] for j in range(len(users))]
mean_ratings=[]





for i in range(len(users)):
  user_data=train_data[train_data['user_id']==users[i]]
  user_song=user_data['song'].unique()
  mean=(user_data['rating']).mean()
  #print("ratings of user  ",i)
  #print(user_data['rating'],mean)
  mean_ratings.append(mean)
  k=0
  for song in user_song:
    user_matrix[i][songs_to_ix[song]]=float(train_data.iloc[temp+k,8])#-mean
    if np.isnan(user_matrix[i][songs_to_ix[song]]):
      user_matrix[i][songs_to_ix[song]]=0
    k+=1
  temp+=len(user_data)


print(len(user_matrix),len(user_matrix[0]))

def collab(user_matrix):
  #print(user_matrix[user_no])
  rating=[]
  for j in range(len(users)):
    #user_data=train_data[train_data['user_id']==users[j]]
    rats=[]
    for i in range(len(user_matrix)):
      z=float(dot( user_matrix[j],user_matrix[i])/(norm(user_matrix[j])* norm(user_matrix[i])))
      if np.isnan(z):
        rats.append(0)
      else:  
        rats.append(z)
      #if(cosine(user_matrix[j],user_matrix[i])==2):
       # print(user_matrix.iloc[j],user_matrix.iloc[i])
    rating.append(rats)
  return rating


  

print("starting to calculate similarity matrix ")
rating=collab(user_matrix)

with open("similarity",'wb') as file:
  pickle.dump(list(rating),file)

with open("similarity",'rb') as file:
  d=pickle.load(file)
 
for i in d:
  print(i)

