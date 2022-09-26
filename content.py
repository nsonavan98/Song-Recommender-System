import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random



class content_based:
# initialise class characteristics
  def __init__(self,data):
    self.songs=[]
    self.recommend=[]
    self.songs_to_ix = { s:i for i,s in enumerate(data['song']) }
    self.ix_to_songs = { i:s for i,s in enumerate(data['song']) }

# Find songs similar to given set using content based filtering
  def recommend_songs(self,songs,all_songs,number):
    i=0
    for s in songs:
      song_data=all_songs[all_songs['song']==s]
      song_data=song_data.drop_duplicates('title')
      if(len(song_data)==0):
        l=s.split('-')
        song_data=all_songs[all_songs['artist_name']==s]
        song_data=song_data.drop_duplicates('artist_name')
      if ( len(song_data)==0 ):
        l=s.split('-')
        song_data=all_songs[all_songs['title']==s]
        song_data=song_data.drop_duplicates('title')

      for i in range(len(all_songs)):
        score=0
        
        for j in range(len(song_data)):  
          if( song_data.iloc[j,2]==all_songs.iloc[i,2] ):
            score+=3

          if(song_data.iloc[j,3]==all_songs.iloc[i,3] ):
            score+=5
          if(song_data.iloc[j,4]==all_songs.iloc[i,4] ):
            score+=1
          if( song_data.iloc[j,1]==all_songs.iloc[i,1] ): 
            score=0

        self.songs.append(score)

      song_ranking=np.argsort(self.songs)

      for j in range(len(song_data)):
        for i in range(number):
          self.recommend.append([self.ix_to_songs[song_ranking[len(song_ranking)-i-1]],self.songs[song_ranking[len(song_ranking)-i-1]]])

    list=[]
    rats=[]

    for i in range(number):
        
        n=0
        while((self.recommend[n][0] in list)):
            n=random.randint(0,len(self.recommend)-1)

        list.append(self.recommend[n][0])  
        rats.append(self.recommend[n][1])
    
    return list,rats
   

  

