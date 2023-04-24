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
    #print("start finding rating",len(songs),number)
    i=0
    for s in songs:
      song_data=all_songs[all_songs['song']==s]
      #print(song_data)
      song_data=song_data.drop_duplicates('title')
      if(len(song_data)==0):
        l=s.split('-')
        #print(l)
        song_data=all_songs[all_songs['artist_name']==s]
        song_data=song_data.drop_duplicates('artist_name')
      if ( len(song_data)==0 ):
        l=s.split('-')
        #print(l)
        song_data=all_songs[all_songs['title']==s]
        song_data=song_data.drop_duplicates('title')
      #i+=1
      #song_data=all_songs[all_songs['title']==l[0]]
      
      #print("indi",song_data)
      for i in range(len(all_songs)):
        score=0
        
        for j in range(len(song_data)):  
          if( song_data.iloc[j,2]==all_songs.iloc[i,2] ):
            score+=3
          #print(song_data.iloc[0,2],i)
          if(song_data.iloc[j,3]==all_songs.iloc[i,3] ):
            score+=5
          if(song_data.iloc[j,4]==all_songs.iloc[i,4] ):
            score+=1
          if( song_data.iloc[j,1]==all_songs.iloc[i,1] ): 
            score=0
        #if(random.randint(0,100)%2==0):
        self.songs.append(score)
         # else:
          #  self.songs.append(0)
      #temp=pd.Series(songs,index=range(len(all_songs)))
      #temp=temp.argsort()
      #song_ranking=self.songs 
      song_ranking=np.argsort(self.songs)
    #print("end\nstart finding songs")
      #song_data=all_songs['song']
    #print(song_data[1])
    #print(song_data[1321])
     # print(len(song_ranking))
      #print(len(song_data))
      for j in range(len(song_data)):
        for i in range(number):
      #if(not (ix_to_songs[song_ranking[len(song_ranking)-i-1]] in history)):
          self.recommend.append([self.ix_to_songs[song_ranking[len(song_ranking)-i-1]],self.songs[song_ranking[len(song_ranking)-i-1]]])
    """for j in range(20):
      next_top=0
      
      for i in range(len(song_data)):
        
        if(song_ranking[next_top]<song_ranking[i]):
          next_top=i
     
      self.recommend.append(song_data[next_top])
      print(song_ranking[next_top])
      song_ranking[next_top]=0"""
    list=[]
    rats=[]
    #lis=[-1]
    #print("all ",self.recommend)
    for i in range(number):
        
        n=0
        while((self.recommend[n][0] in list)):
            n=random.randint(0,len(self.recommend)-1)
            #print(n)
            
        #lis.append(n)
        #print(n)
        list.append(self.recommend[n][0])  
        rats.append(self.recommend[n][1])
    #list=[self.recommend[random.randint(0,len(self.recommend))] for i in range(number)]
    
    return list,rats
   
"""song_db_2=pd.read_csv('/home/aditya/Downloads/song_data.csv')
song_db_2['song']=song_db_2['title']+"-"+song_db_2['artist_name']




song=['Justin Bieber']
        
content=content_based()
recom=content.recommend_songs(song,song_db_2)      

for r in recom:
  print (r)"""
  

