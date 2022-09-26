import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm


class item_based:

    def __init__(self,train_data,song_db_subset,songs_to_ix,ix_to_songs):
        self.train_data=train_data
        self.song_db_subset=song_db_subset
        self.users=song_db_subset['user_id'].unique()
        self.songs=song_db_subset['song'].unique()
        self.user_matrix=[[0 for i in range(len(self.users))] for j in range(len(self.songs))]
        self.mean_ratings=[]
        self.similarity=[]
        self.pred=[]
        
        temp=0

       
        for i in range(len(self.users)):
            user_data=train_data[train_data['user_id']==self.users[i]]
            user_song=user_data['song'].unique()
            mean=(user_data['rating']).mean()
            self.mean_ratings.append(mean)
            k=0
            for song in user_song:
                self.user_matrix[songs_to_ix[song]][i]=float(train_data.iloc[temp+k,8])
                if np.isnan(self.user_matrix[songs_to_ix[song]][i]):
                    self.user_matrix[songs_to_ix[song]][i]=0
                k+=1
            temp+=len(user_data)
       
    
    def collab(self):
        alpha=0.1
        for j in range(len(self.songs)):
            rats=[]
            for i in range(len(self.user_matrix)):
                d=( (norm(self.user_matrix[j])**alpha)* (norm(self.user_matrix[i])**(1-alpha)) )
                if d==0:
                    rats.append(0)
                else:
                    rats.append( dot( self.user_matrix[j],self.user_matrix[i]) / d ) 
            self.similarity.append(rats)
        return self.similarity.copy()
        
        
    def predict_ratings(self,user_no):
    
        song=[]
        self.pred=[]
        for i in range(len(self.songs)):
            if(self.user_matrix[i][user_no]==0  ):
                sum_sim_rat=0
                sum_sim=0
                for j in range(len(self.user_matrix)):
                    if(self.user_matrix[j][user_no]!=0):
                        sum_sim_rat+=(self.similarity[i][j]*self.user_matrix[j][user_no])
                        #print(sum_sim_rat,self.user_matrix[j][user_no])
                        sum_sim+=abs(self.similarity[i][j])
    
                if(sum_sim_rat != 0 and sum_sim>0):
                    #print(sum_sim_rat,sum_sim)
                    self.pred.append(sum_sim_rat/sum_sim )

                    if( not np.isnan(self.pred[i])):
                         song.append(self.songs[i])
                else:
                    self.pred.append(self.mean_ratings[user_no])
            else:
                self.pred.append(0)
        #print(self.pred)
        return self.pred.copy(),song
         
    def  predict_songs(self,number,x,ix_to_songs,songs_to_ix,user_no,ub_songs):
        songs=[]
        error=0
        number=int((x/100)*number)
        #print(self.similarity)
        user_test=self.train_data[self.train_data['user_id']==self.users[user_no]]
        history=user_test['song']
        
        user_test=self.train_data[self.train_data['user_id']==self.users[user_no]]
        history=user_test['song']
        
        """if(len(user_test)!=0):
            for j in range(len(user_test)):
               # if self.pred[songs_to_ix[user_test.iloc[j,12]]] !=self.mean_ratings[user_no]:
                   # print("ATTENTION  ",user_test.iloc[j,8],self.pred[songs_to_ix[user_test.iloc[j,12]]])
                error+=abs(user_test.iloc[j,8]-self.pred[songs_to_ix[user_test.iloc[j,12]]])

            print("error",user_no,"-",error/len(user_test))
        #error/=len(user_test)"""
        
        
        rank=np.argsort(self.pred)
        #print(self.pred)
        #print(rank)
        rats = []
        no = 0
        for i in range(len(rank)):
            if no == number:
                break
            
            if(not (ix_to_songs[rank[len(rank)-i-1]] in history or ix_to_songs[rank[len(rank)-i-1]] in ub_songs) ):
                songs.append(ix_to_songs[rank[len(rank)-i-1]])
                rats.append(self.pred[rank[len(rank)-i-1]])
                no+=1
                
        return songs,rats       
               
 
