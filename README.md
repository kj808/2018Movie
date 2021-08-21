# Movie

Description: A variety of movies exist and with today's technology, it is easier to binge watch movies catered towards personal tastes. With this knowledge of different genres, tags and ratings, is it possible to cluster similar movies?

Location:  Data Science Graduate Course at KU 

Dates: Sep 26, 2018 to Oct 5, 2018

## Goal
* Cluster similar movies

## Data
* [A list of movies](https://grouplens.org/datasets/movielens/). Movies include an ID, title, year, and genre.

## Analysis
* I performed data rangling on the two data sets (i.e., normalized features, combined data sets, removing unrelevant entries, randomizing entries, transforming features for model via bag of words)
* I created two clustering models: K-means and DBScan
* Visualized the clusters via matplotlib

## Results
Based on the data setup, it is difficult to understand the results. Genres and tags are split into multiple columns and contain binary data for each instance. With the variety of columns, plotting is difficult to understand. In terms of algorithms, Kmeans is set to four clusters based on DBScan's output of 3 clusters. All in all, altering the input data will assist with better understandable output.

## Repository Contents

| Directory | Description |
| --- | ----------- |
| Data | Contains all of the datasets used in this project. |
| Libraries | If libraries are used, the exact distribution will be located here. Includes library, library name, and library version. |
| Models | Models generated for the project such as machine learning models. |
| Notebooks | Notebooks used for visualing the data. |
| Reports | The resulting reports on this project. |
| Src | Source scripts and other helper files located here. |

