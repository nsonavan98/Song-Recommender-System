import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm


class user_based:

# initialise all the class characteristics
    def __init__(self,train_data,song_db_subset,songs_to_ix,ix_to_songs):
        self.train_data=train_data
        self.song_db_subset=song_db_subset
        self.users=song_db_subset['user_id'].unique()
        self.songs=song_db_subset['song'].unique()
        self.user_matrix=[[0 for i in range(len(self.songs))] for j in range(len(self.users))]
        self.mean_ratings=[]
        self.similarity=[]
        self.prediction=[]

#   create user matrix where rows are users and columns are all songs in dataset 
#   and each cell is rating given by a particular user to a particular song  
        user_offset=0
        for i in range(len(self.users)):
            user_data=train_data[train_data['user_id']==self.users[i]]
            user_song=user_data['song'].unique()
            mean=(user_data['rating']).mean()
            self.mean_ratings.append(mean)
            song_offset=0
            for song in user_song:
                self.user_matrix[i][songs_to_ix[song]]=float(train_data.iloc[user_offset+song_offset,8])
                song_offset+=1
            user_offset+=len(user_data)
    
#   find similarity between users by cosine similarity with little variation
#   if Single is not False calculate similarity between given user and all other in maatrix 
#   else find simialrity of each user with another   
    def collab(self,Single=False):
        alpha=0.15
        if Single == False:
            for j in range(len(self.users)):
                rats=[]
                for i in range(len(self.user_matrix)):
                    d=((norm(self.user_matrix[j])**alpha)* (norm(self.user_matrix[i])**(1-alpha)))
                    if d==0:
                        rats.append(0)
                    else:
                        rats.append( dot( self.user_matrix[j],self.user_matrix[i]) / d ) 
                self.similarity.append(rats)
        else:
            rats=[]
            for i in range(len(self.user_matrix)):
                d=((norm(self.user_matrix[Single])**alpha)* (norm(self.user_matrix[i])**(1-alpha)))
                if d==0:
                    rats.append(0)
                else:
                    rats.append( dot( self.user_matrix[Single],self.user_matrix[i]) / d )
            self.similarity.append(rats)
        return self.similarity.copy()
        
# calculate ratings for all songs unrated by given user using pearson correlation    
    def predict_ratings(self,user_no,K,Single=False):
        song=[]
        self.prediction=[]
        
        u = 0
        if Single == False:
            u=user_no

        rank = np.argsort(self.similarity[u])
        for i in range(len(self.songs)):
            if(self.user_matrix[user_no][i]==0  ):
                sum_sim_rat=0
                sum_sim=0

                
                for l in range(min(K,len(self.similarity[0]))):
                    j = rank[len(rank) - l - 1]
                    if(self.user_matrix[j][i]!=0):
                        sum_sim_rat+=self.similarity[u][j]*self.user_matrix[j][i]
                        sum_sim+=abs(self.similarity[u][j])
                     
                if(sum_sim_rat != 0 and sum_sim_rat>0):
                    self.prediction.append(sum_sim_rat/sum_sim )

                    if( not np.isnan(self.prediction[i])):
                         song.append(self.songs[i])
                else:
                    self.prediction.append(self.mean_ratings[user_no])
            else:
                self.prediction.append(0)
        return self.prediction.copy(),song

# Make song prediction      
    def  predict_songs(self,number,x,ix_to_songs,songs_to_ix,user_no):
        songs=[]
        number=int((x/100)*number)
        error=0
        user_test=self.train_data[self.train_data['user_id']==self.users[user_no]]
        history=user_test['song']
        
        rank=np.argsort(self.prediction)
        rats=[]

        for i in range(number):
            if(not (ix_to_songs[rank[len(rank)-i-1]] in history)):
                songs.append(ix_to_songs[rank[len(rank)-i-1]])
                rats.append(self.prediction[rank[len(rank)-1-i]])
                
        return songs,rats       
               
 
