import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import user_class
import item_class
import content
import Popularity
from evalaution import precision_recall_calculator as prc


def user(user_id,no=20):
    global songs_to_ix,ix_to_songs,id_to_no,no_to_id,train_data,test_data,users,songs,song_db_subset
    User_no = id_to_no[user_id]

    ub_obj=user_class.user_based(train_data,song_db_subset,songs_to_ix,ix_to_songs)

    #calculating the user user similarity
    ub_obj.collab(User_no)
    #print(ub_obj.similarity)

    #calculating user ratings for that given user
                                #  KNN
    ub_obj.predict_ratings(User_no,20,True)

    songs,rats=ub_obj.predict_songs(no,100,ix_to_songs,songs_to_ix,User_no)

    return songs,rats

def item(user_id,no=20):
    global songs_to_ix,ix_to_songs,id_to_no,no_to_id,train_data,test_data,users,songs,song_db_subset,item_obj
    User_no = id_to_no[user_id]

    item_obj.predict_ratings(User_no)

    songs,rats=item_obj.predict_songs(no,100,ix_to_songs,songs_to_ix,User_no,songs)

    return songs,rats


def popularity(user_id,no=20):
    global songs_to_ix,ix_to_songs,id_to_no,no_to_id,train_data,test_data,users,songs,song_db_subset
    User_no = id_to_no[user_id]

    song_db_1=pd.read_csv('/home/niket/Desktop/triplet.csv')
    song_db_2=pd.read_csv('/home/niket/Downloads/song_data.csv',)
    song_db=pd.merge(song_db_1,song_db_2.drop_duplicates(['song_id']),on="song_id",how="left")
    song_db['song']=song_db['title']+"-"+song_db['artist_name']
    song_grouped=song_db.groupby(['song']).agg({'listen_count':'count'}).reset_index()
    grouped_sum=song_grouped['listen_count'].sum()
    
    train_data,test_data=train_test_split(song_db_subset,test_size=.20,random_state=0)
    #print(train_data)
    pm=Popularity.popularity_recommender()
    
    pm.make(train_data,'user_id','song',no)
    return (list(pm.recommend_songs(user_id)['song']))






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

id_to_no = { id:i for i,id in enumerate(users)}
no_to_id = { i:id for i,id in enumerate(users)}

item_obj=item_class.item_based(train_data,song_db_subset,songs_to_ix,ix_to_songs)
    
    #calculating the item item similarity
item_obj.collab()
    
    #calculating user ratings for that given user

#print(user("969cc6fb74e076a68e36a04409cb9d3765757508"))

#print(item("969cc6fb74e076a68e36a04409cb9d3765757508"))

#print(popularity("969cc6fb74e076a68e36a04409cb9d3765757508"))

p = prc(test_data,train_data,popularity,user,item)

l = p.calculate_measures(1)
print("\n\n**********************************************************************\n")
print("popularity precision")
p = np.asarray(l[0]).mean()
print(np.asarray(l[0]).mean())
#print(l[0])
print("popularity recall")
r = np.asarray(l[1]).mean()
print(np.asarray(l[1]).mean())
#print(l[1])
print("F1 score")
print(2*(p*r)/(p+r))
print("\n\n")
print("user precision")
p = np.asarray(l[2]).mean()
print(np.asarray(l[2]).mean())
#print(l[2])
print("user recall")
r = np.asarray(l[3]).mean()
print(np.asarray(l[3]).mean())
#print(l[3])
print("F1 score")
print(2*(p*r)/(p+r))
print("\n\n")
print("item precision")
p = np.asarray(l[4]).mean()
print(np.asarray(l[4]).mean())
#print(l[4])
print("item recall")
r = np.asarray(l[5]).mean()
print(np.asarray(l[5]).mean())
#print(l[5])
print("F1 score")
print(2*(p*r)/(p+r))
print("\n\n")
print("user-item precision")
p = np.asarray(l[6]).mean()
print(np.asarray(l[6]).mean())
#print(l[6])
print("user-item recall")
r = np.asarray(l[7]).mean()
print(np.asarray(l[7]).mean())
#print(l[7])
print("F1 score")
print(2*(p*r)/(p+r))
print("\n\n")
print("user mae")
print(np.asarray(l[8]).mean())
#print(l[8])
print()
print("item mae")
print(np.asarray(l[9]).mean())
#print(l[9])
print()
print("user-item mae")
print(np.asarray(l[10]).mean())
#print(l[10])
