import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import user_class
import item_class
import content
import Popularity


#get the data and preposing

#for collab
song_db=pd.read_csv("/home/niket/Desktop/rating1.csv")

#for content
song_db_2=pd.read_csv('/home/niket/Downloads/song_data.csv')
song_db_2['song']=song_db_2['title']+"-"+song_db_2['artist_name']



#taking only subset of data
song_db_subset=song_db.head(1000).copy()
song_db_subset['rating']=song_db_subset['rating'].apply(pd.to_numeric, errors='coerce')

#split data into train and test
train_data,test_data=train_test_split(song_db_subset,test_size=.20,random_state=0)
users=song_db_subset['user_id'].unique()
songs=song_db_subset['song'].unique()

print("Number of unique users: ",len(users) )
print("Number of unique songs: ",len(songs) )

#song to number and number to song encodings

songs_to_ix = { s:i for i,s in enumerate(songs) }
ix_to_songs = { i:s for i,s in enumerate(songs) }


#user no in the dataset for which recommendations are to be given
User_no = 2
No=20 #no of songs to recommend
x=50 #percentage of item based filtering to use

User=users[User_no]

user_data=train_data[train_data['user_id']==users[User_no]]

print("no of songs in history of user",len(user_data['song'].unique()))
#User=123
print("\n\n**********************************************************************\n")
print("Recommending songs for user:",users[User_no])
print("\n\n**********************************************************************\n")
if User not in users or len(user_data['song'].unique()) == 0:
    #for popularity
    print("Popularity based model")
    song_db_1=pd.read_csv('/home/niket/Desktop/triplet.csv')
    song_db_2=pd.read_csv('/home/niket/Downloads/song_data.csv',)
    song_db=pd.merge(song_db_1,song_db_2.drop_duplicates(['song_id']),on="song_id",how="left")
    song_db['song']=song_db['title']+"-"+song_db['artist_name']
    song_grouped=song_db.groupby(['song']).agg({'listen_count':'count'}).reset_index()
    grouped_sum=song_grouped['listen_count'].sum()
    
    train_data,test_data=train_test_split(song_db_subset,test_size=.20,random_state=0)
    #print(train_data)
    pm=Popularity.popularity_recommender()
    
    pm.make(train_data,'user_id','song',No)
    print(pm.recommend_songs(User))
    #l=list(pm.recommend_songs(User)['song'])
    #for i in l:
    #    print(i)
    
else:




#content based filtering
    song=[]
    
    
    if(len(user_data['song'].unique()) < 10):
        print("20 % Content based filtering \n")
        #print("80 % 20% content \n")
    
        for i in range(len(user_data)):
          if(user_data.iloc[i,8]>3):
            song.append(user_data.iloc[i,12]) 
        content=content.content_based(song_db_2)
        #song=['Yellow']
        recom,score=content.recommend_songs(song,song_db_2,int(No/5))      
        d = {'song':recom,'score':score}
        song_list=pd.DataFrame(d,index=[i for i in range(1,len(recom)+1)])
        print(song_list)
        #for r in recom: 
         #   print (r)
    
        No=int(No*4/5)
        print("\n\n**********************************************************************\n")
        print("80% ",end='')
    #userbased collaborative filtering
    ub_obj=user_class.user_based(train_data,song_db_subset,songs_to_ix,ix_to_songs)

    #calculating the user user similarity
    ub_obj.collab(User_no)
    #print(ub_obj.similarity)

    #calculating user ratings for that given user
    ub_obj.predict_ratings(User_no,20,True)
    #print(ub_obj.pred)
    #getting recommendations
    
    
    print("User based recommendation\n")
    
    
    songs,rats=ub_obj.predict_songs(No,100-x,ix_to_songs,songs_to_ix,User_no)
    d = {'song':songs,'score':rats}
    list_song = pd.DataFrame(d,index=[i for i in range(1,len(songs)+1)])
    print(list_song)
    #for i in range(len(songs)):
    #    print(list_song.iloc[i])
        
    print("\n\n**********************************************************************\n")
    
    print("Item based recommendation\n")
    #item based collaborative filtering
    item_obj=item_class.item_based(train_data,song_db_subset,songs_to_ix,ix_to_songs)
    
    #calculating the item item similarity
    item_obj.collab()
    
    #calculating user ratings for that given user
    item_obj.predict_ratings(User_no)
    
    
    
    songs,rats=item_obj.predict_songs(No,x,ix_to_songs,songs_to_ix,User_no,songs)
    d =  {'song':songs,'score':rats}
    list_song = pd.DataFrame(d,index=[i for i in range(1,len(songs)+1)])
    print(list_song)
    #for i in songs:
    #    print(i)
        
    print("\n\n**********************************************************************\n")
