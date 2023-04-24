import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm


#get the data and preposing
song_db=pd.read_csv("/home/aditya/Desktop/rating1.csv")
#song_db['song']=song_db['title']+"-"+song_db['artist_name']
#print(song_db.head(10))


    





song_db_subset=song_db.head(1000).copy()
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






def collab(user_matrix):
  #print(user_matrix[user_no])
  rating=[]
  """for j in range(len(users)):
    #user_data=train_data[train_data['user_id']==users[j]]
    rats=[]
    for i in range(len(user_matrix)):
      rats.append( dot( user_matrix[j],user_matrix[i])/(norm(user_matrix[j])* norm(user_matrix[i])) )
      #if(cosine(user_matrix[j],user_matrix[i])==2):
       # print(user_matrix.iloc[j],user_matrix.iloc[i])
    rating.append(rats)"""
  rating=cosine_similarity(user_matrix)
  return rating



def predict_users(rating,similarity,user_no):
  song=[]
  rating_temp=[]
  
  for i in range(len(songs)):
    if(rating[user_no][i]==0  ):
      sum_sim_rat=0
      sum_sim=0
      for j in range(len(rating)):
        if(rating[j][i]!=0):
          sum_sim_rat+=similarity[user_no][j]*rating[j][i]
          sum_sim+=abs(similarity[user_no][j])
    
      if(sum_sim_rat != 0 and sum_sim_rat>0):
        rating_temp.append(sum_sim_rat/sum_sim )

        
        if( not np.isnan(rating_temp[i])):
          song.append(songs[i])
      else:
        rating_temp.append(mean_ratings[user_no])
        #print(song)
    else:
      rating_temp.append(0)
  return rating_temp,song
 
u=[]

  
#print("user rating matrix  ")
#for i in u:
#  print(i)
rating=collab(user_matrix)

#print("user similarity matrix  ")
#for r in rating:
#  print(r)



total_error=0.0
n=0
print(len(rating[0]),len(songs))
for user_no in range(len(users)):
 
  

  #user_similarity = pairwise_distances(user_matrix, metric='cosine')
  pred,song=predict_users(user_matrix,rating,user_no)

 # print("predicted user matrix ratings ",pred)
  #for i in pred: 
  #  print(i)





  #print(pred)

  user_test=test_data[test_data['user_id']==users[user_no]]
  pred_index=[]
  #print("songs of user in test set \n",(user_test['song']))
  
  #print("calculated rating songs")
  #for i in song: 
  #  print(i)
  #print(len(song))
  actual_index=[]
  error=0.0
  precision=0.0
  if(len(user_test)!=0):
    for j in range(len(user_test)):
      #if pred[songs_to_ix[user_test.iloc[j,12]]] !=mean_ratings[user_no]:
        #print("ATTENTION  ",user_test.iloc[j,8],pred[songs_to_ix[user_test.iloc[j,12]]])
      error+=abs(user_test.iloc[j,8]-pred[songs_to_ix[user_test.iloc[j,12]]])

    print("error",user_no,"-",error)
    error/=len(user_test)
    print("avg error",user_no,"-",error)
    if(error!=0 and not np.isnan(error)):
      total_error+=error
      print("cummulative total error",total_error)
      n+=1
   

    recom=[]
    """for i in range(10):
      max=0
      k=0
      for j in range(len(songs)):
        if(max<pred[j] and songs[j] not in user_test['song']):
          k=j
          max=pred[j]
      pred[k]=0
      recom.append(k)

    print()
    print("recommend songs")
    print()
    for i in recom:
      print(i)
    """
    
    
    #pecision
    
    actual_relevant=0
    actual_rev_list=[]
    pred_relevant=0
    pred_rev_list=[]
    
    for i in range(len(user_test)):
      if( user_test.iloc[i,8] > 3 ):
        actual_relevant+=1
        actual_rev_list.append(user_test.iloc[i,12])
    #print("relevant songs number ",actual_relevant)    
    sorted_rat=np.argsort(pred)
    length=len(sorted_rat)
    #print(length)
    precision_list=[]
    
    for i in range(100):
      if(pred[sorted_rat[length-i-1]] > 3 and (ix_to_songs[sorted_rat[length-i-1]] in actual_rev_list)):
        pred_relevant+=1
        pred_rev_list.append(sorted_rat[length-i-1])
      precision_list.append(pred_relevant/(i+1))
    #print("precision are ",precision_list)  
    avg_precision=np.mean(precision_list)
    print("average precision ",avg_precision)  
    precision+=avg_precision
          
    
    
    
    
    
    
    
    
    
    
    
    
print("final total error",total_error,n)
print("final total precision",precision)
total_error/=n
print("avg final total error",total_error)
print("avg final precision ",precision/n)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  #"""for i in user_test['song']:
   # if i in song:
    #  print("yes")
    #else:
     # print("no")"""

"""for i in song:
  if i in list(user_test['song']):
    print("yes")
  else:
    print("no")"""
"""for i in user_similarity:
  print(i)
TTENTION 1.36124401914 1.0
ATTENTION 5.36898395722 1.0
ATTENTION 2.1993006993 1.0
ATTENTION 2.37914429795 1.0
ATTENTION 1.29545454545 1.0
ATTENTION 1.72285546009 1.0
ATTENTION 7.5089891334 3.0
ATTENTION 7.5089891334 5.0
ATTENTION 7.56769547546 1.0
ATTENTION 4.75297133346 3.0
ATTENTION 5.45029365844 3.0
ATTENTION 6.32694829823 1.0
ATTENTION -0.6666666666666665 2.0
ATTENTION -0.1212121212121211 1.0
ATTENTION 0.671875 1.0
ATTENTION -0.328125 1.0

for i in rating :
  print(i)"""
