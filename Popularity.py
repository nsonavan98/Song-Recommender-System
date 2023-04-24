import numpy as np
import pandas as pd
#import scipy as s
from scipy import sparse
#from .base import clone
import math
import csv


class popularity_recommender():

# initialise class characteristics

  def __init__(self):
    self.train_data=None
    self.songs=None
    self.user_id=None
    self.recommendions=None
  
# make dataframe of songs sorted based on no of times its been listened to
  def make(self,train_data,user_id,song,number):
    self.train_data=train_data
    self.user_id=user_id
    self.songs=song
    
    song_hits=self.train_data.groupby([self.songs]).agg({self.user_id:'count'}).reset_index()
    song_hits.rename(columns={'user_id':'score'},inplace=True)
    song_hits_sort=song_hits.sort_values(['score',self.songs],ascending=[0,1]) 
    
    song_hits_sort['rank']=song_hits_sort['score'].rank(ascending=0,method='first')
    
    self.recommendations=song_hits_sort.head(number)
    

# make recommendation for user   
  def recommend_songs(self,user_id):
    user_recom=self.recommendations
    user_recom['user_id']=user_id
    
    columns=user_recom.columns.tolist()
    #print(columns,columns[-1:],columns[:-1])
    columns=['rank','song','score']
    user_recom=user_recom[columns]
    
    return user_recom
    
