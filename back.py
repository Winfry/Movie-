import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (replace 'your_data.csv' with your actual file)
"""
    read csv file
    :param path:file path
    :return:csv data as DataFrame
    """
data = pd.read_csv("data(1).txt")
data.to_csv('data.csv', index=None )
   

# Pivot the data to create a user-movie matrix
user_movie_matrix = data.pivot_table(index='User', columns='Movie', values='Rating', fill_value=0)

# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)

# Function to get movie recommendations for a given movie
def get_movie_recommendations(movie_name, similarity_matrix, user_movie_matrix):
    similar_scores = similarity_matrix[movie_name]
    ranked_movies = list(user_movie_matrix.columns)
    ranked_movies.remove(movie_name)

    recommendations = pd.DataFrame(similar_scores, index=ranked_movies, columns=['Similarity'])
    recommendations = recommendations.sort_values(by='Similarity', ascending=False)

    return recommendations

# Command-line interface
while True:
    movie_input = input("Enter a movie name (type 'exit' to quit): ").strip().title()

    if movie_input.lower() == 'exit':
        print("Exiting the recommendation system. Goodbye!")
        break

    if movie_input in user_movie_matrix.columns:
        recommendations = get_movie_recommendations(movie_input, movie_similarity, user_movie_matrix)
        print(f"\nTop 5 movie recommendations for '{movie_input}':\n")
        print(recommendations.head(5))
    else:
        print(f"Movie '{movie_input}' not found in the dataset. Please enter a valid movie name.")
