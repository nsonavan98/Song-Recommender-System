import math as mt
import csv
from sparsesvd import sparsesvd #used for matrix factorization
import numpy as np
from scipy.sparse import csc_matrix #used for sparse matrix
from scipy.sparse.linalg import * #used for matrix multiplication
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm





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
    k+=1
  temp+=len(user_data)


print(len(user_matrix),len(user_matrix[0]))




MAX_PID = len(songs)
MAX_UID = len(users)

#Compute SVD of the user ratings matrix
def computeSVD(urm, K):
    U, s, Vt = sparsesvd(urm, K)
    #print(U)
    #print()
   # print(s)
  #  print()
 #  print(Vt)
 #   print() 
    dim = (len(s), len(s))
    print("dim is  ",dim)
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
   # print(U)
    #print()
    #print(S)
    #print()
    #print(Vt)
    #print() 
    return U, S, Vt

#Compute estimated rating for the test user
def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
    """print(U)
    print()
    print(S)
    print()
    print(Vt)
    print()"""
    rightTerm = S*Vt 
    r=rightTerm.todense()
    #print(r)
    #print()
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    #for userTest in uTest:
    prod = U[uTest, :]*rightTerm
        #print(prod)
    print()
        #we convert the vector to dense format in order to get the indices 
        #of the movies with the best estimated ratings 
    estimatedRatings[uTest, :] = prod.todense()
    print(estimatedRatings[uTest])
        
       # recom = (-estimatedRatings[userTest, :]).argsort()
    return estimatedRatings[uTest]


K=10

urm = np.array(user_matrix)
urm = csc_matrix(urm, dtype=np.float32)

U, S, Vt = computeSVD(urm, K)


total_error=0.0
n=0
for user_no in range(len(users)):
  pred = computeEstimatedRatings(urm, U, S, Vt,(user_no), K, True)
  
  user_test=test_data[test_data['user_id']==users[user_no]]
  pred_index=[]
  print("songs of user in test set \n",(user_test['song']))
  error=0.0
  precision=0.0
  if(len(user_test)!=0):
    for j in range(len(user_test)):
      if pred[songs_to_ix[user_test.iloc[j,12]]] !=mean_ratings[user_no]:
        print("ATTENTION  ",user_test.iloc[j,8],pred[songs_to_ix[user_test.iloc[j,12]]])
      error+=abs(user_test.iloc[j,8]-pred[songs_to_ix[user_test.iloc[j,12]]])

    print("error",user_no,"-",error)
    error/=len(user_test)
    print("avg error",user_no,"-",error)
    if(error!=0 and not np.isnan(error)):
      total_error+=error
      print("cummulative total error",total_error)
      n+=1
    actual_relevant=0
    actual_rev_list=[]
    pred_relevant=0
    pred_rev_list=[]
    
    for i in range(len(user_test)):
      if( user_test.iloc[i,8] > 3 ):
        actual_relevant+=1
        actual_rev_list.append(user_test.iloc[i,12])
    print("relevant songs number ",actual_relevant)    
    sorted_rat=np.argsort(pred)
    length=len(sorted_rat)
    print(length)
    precision_list=[]
    
    for i in range(20):
      if(pred[sorted_rat[length-i-1]] > 3 and (ix_to_songs[sorted_rat[length-i-1]] in actual_rev_list)):
        pred_relevant+=1
        pred_rev_list.append(sorted_rat[length-i-1])
      precision_list.append(pred_relevant/(i+1))
    print("precision are ",precision_list)  
    avg_precision=np.mean(precision_list)
    print("average precision ",avg_precision)  
    precision+=avg_precision
          
    
    
    
    
    
    
    
    
    
    
    
    
print("final total error",total_error,n)
print("final total precision",precision)
total_error/=n
print("avg final total error",total_error)
print("avg final precision ",precision/n)

