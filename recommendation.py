import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return data[data.index == index]["Movie"].values[0]

def get_index_from_title(Movie):
	return data[data.Movie == Movie]["Movie"].values[0]

def get_rating_from_index(index):
	return data[data.index == index]["Rating"].values[0]

def get_rating_from_user(Rating):
	return data[data.Rating == Rating]["index"].values[0]

##################################################

##Step 1: Read CSV File
"""
    read csv file
    :param path:file path
    :return:csv data as DataFrame
    """
data = pd.read_csv("data(1).txt")
data.to_csv('data.csv', index=None )
   
##Step 2: Select Features

features = ['Rating', 'Movie', 'User']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	data[feature] = data[feature].fillna('')

def combine_feature(row):
	try:
		return row['Rating'] +' '+ row["User"] + " " + row['Movie'] 
	except:
		print('Error : \n', row)
data['combined_features'] = data.apply(combine_feature, axis=1)

#print(df['combined_features'].head())
##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
cv_fit=cv.fit_transform(data['combined_features'])

##Step 5: Compute the Cosine Similarity based on the count_matrix
similarity = cosine_similarity(cv_fit )
high_ratings = input("Enter highest Rated Movies:")

## Step 6: Get index of this movie from its title
rating_index = get_rating_from_index(high_ratings)
try:
    movie_index = int(rating_index)
    similar_movies = list(similarity[movie_index])
except IndexError:
    print("Invalid movie index.")
except ValueError:
    print("Invalid movie index format. Must be an integer.")



## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies, key= lambda x: x[1], reverse = True)

## Step 8: Print titles of first 50 movies
i = 0
for mo in sorted_similar_movies:
	print(get_title_from_index(mo[0]))
	i = i+ 1
	if i>50:
		break