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
import time


t=time.time
a=[[1,0,1],[1,2,1],[0,0,0]]

rating=[]
for j in range(len(a)):
    #user_data=train_data[train_data['user_id']==users[j]]
    rats=[]
    for i in range(len(a)):
      rats.append( dot( a[j],a[i])/(norm(a[j])* norm(a[i])) )
      #if(cosine(user_matrix[j],user_matrix[i])==2):
       # print(user_matrix.iloc[j],user_matrix.iloc[i])
    rating.append(rats)
print(rating)
print(cosine_similarity(a))
print(pairwise_distances(a))

print([1,2]+[3,5])

print( {2 , 3 ,5 ,7}.intersection({ 3,9,6,7}) )