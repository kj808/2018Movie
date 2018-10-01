#Import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # KMEANS clustering
from sklearn.cluster import DBSCAN # DBSCAN clustering

#Import datasets
genreSrc=pd.read_csv('../data/external/movies.csv')
ratingSrc=pd.read_csv('../data/external/ratings.csv')
tagSrc=pd.read_csv('../data/external/tags.csv')

#----------9742=genreSrc ----------
#NEED TO DROP TITLES
#Genre data set requires extraction of year and dropping of Title
Year=genreSrc.title.str.extract(r'([a-zA-Z0-9].*)([^(])\(([0-9]*)\)')
Year.columns=['title', 'blank', 'year']

#Drop columns before merge
genreSrc=genreSrc.drop(columns=['title'])
Year=Year.drop(columns=['title', 'blank'])

genreSrc=genreSrc.merge(Year,left_index=True,right_index=True)


#----------100836=ratingSrc ----------
#NEED TO DROP USERID, TIMESTAMP
#Future ideas...(should take mean for each movie :3) 
ratingSrc=ratingSrc.drop(columns=['userId','timestamp'])


#----------3683=tagSrc ----------
#NEED TO DROP USERID AND TIMESTAMP
tagSrc=tagSrc.drop(columns=['userId','timestamp'])

#----------Create mean rating for each movie----------
ratingSrc=ratingSrc.groupby('movieId')['rating'].mean().to_frame()

#Join with genre dataframe
genreSrc=genreSrc.merge(ratingSrc,on='movieId', how='left')

#----------Create an count of tags----------
#Group repeated tags based on movie ID
tagSrc=tagSrc.sort_values(by=['movieId'])

#Count of tags based on movie
tagTotal=tagSrc.groupby(['movieId']).tag.value_counts().to_frame().rename(columns={'tag':'tagCount'})

#Merge with genre such that movies can have multiple tags
#Separate genres into an array
genreSrc['genres']=genreSrc['genres'].str.strip().str.split('[^A-Za-z-]+')

#place all genres separately and into a new dataframe
rows=list()
for row in genreSrc[['movieId','genres']].iterrows():
    r=row[1]
    for genre in r.genres:
        rows.append((r['movieId'], genre))

genreDoc=pd.DataFrame(rows, columns=['movieId', 'genres'])

#Creates count for each genre according to movieId
gDocGroup=genreDoc.groupby(['movieId']).genres.value_counts().to_frame().rename(columns={'genres':'gc'})


#-------Alter Genres to represent bag of words-----------------
gDocGroup=gDocGroup.reset_index()
gDocGroup=gDocGroup.pivot(index='movieId', columns='genres', values='gc')

gDocGroup=gDocGroup.fillna(value=0)

#Drop incorrectly captured genres
gDocGroup=gDocGroup.drop(columns=['genres','listed','no',''])

#-------Alter tags to represent bag of words-----------------
tagTotal=tagTotal.reset_index()
tagTotal=tagTotal.pivot(index='movieId',columns='tag',values='tagCount')
tagTotal=tagTotal.fillna(value=0)

#-------Combine the two-----------------
final=gDocGroup.merge(tagTotal, how='left', left_on='movieId', right_on='movieId')

#--------Add years and ratings-----------
#Insert year and rating to beginning
final.insert(0, 'year', Year['year'])
final.insert(1, 'rating', ratingSrc['rating'])


#Finalize
final=final.fillna(value=0)
final['year']=final['year'].astype(int)
final['rating']=final['rating'].astype(int)

#--------Output to internal-------------
final.to_csv('../data/internal/diction.csv')

#--------Setting up for clustering-------------
#Place into an array
feature_matrix = final.values

#Attach KMEANS
kclustering=KMeans(n_clusters=4)
kclustering.fit(X=feature_matrix)
y_kmeans=kclustering.predict(feature_matrix)

#Attach DBSCAN
DBclustering=DBSCAN(eps=3, min_samples=2).fit(feature_matrix)
y_DBclustering=DBclustering.fit_predict(feature_matrix)

#-------------KMEANS nice-----------------
plt.scatter(feature_matrix[:, 0], feature_matrix[:, 1], c=y_kmeans, s=50, cmap='viridis')

plt.scatter(feature_matrix[:, 1608], feature_matrix[:, 1609], c=y_DBclustering, s=50, cmap='viridis')

