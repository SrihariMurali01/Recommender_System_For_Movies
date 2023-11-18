import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tqdm import tqdm
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df_movies = pd.read_csv('movies.csv').head(10000)
df_ratings = pd.read_csv('ratings.csv').head(1048576)

dfMerged = pd.merge(df_ratings, df_movies, on='movieId')

userEncoder = LabelEncoder()
movieEncoder = LabelEncoder()

dfMerged['userEncoded'] = userEncoder.fit_transform(dfMerged['userId'])
dfMerged['movieEncoded'] = movieEncoder.fit_transform(dfMerged['movieId'])

model = load_model('best_model.h5')

recommended_movies = []

user_id = int(input("Enter the User ID: (1-74): "))

for movie_id in tqdm(df_movies['movieId'], desc="Processing Movie Preferences"):
    user_encoded = userEncoder.transform([user_id])[0]
    try:
        movie_encoded = movieEncoder.transform([movie_id])[0]
    except ValueError:
        continue

    prediction = model.predict([np.array([user_encoded]), np.array([movie_encoded])], verbose=0)[0][0]
    movieTitle = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values[0]
    movie_genres = df_movies.loc[df_movies['movieId'] == movie_id, 'genres'].values[0]

    if prediction >= 2.5:
        recommended_movies.append((movieTitle, prediction, movie_genres.split('|')))

# Aggregating the genres
all_genres = [genre for _, _, genres in recommended_movies for genre in genres] #Extracting all genres
common_genres_counter = Counter(all_genres)


# Sort recommended movies by predicted ratings
sorted_recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)

print(f"\n\nRecommended Movies for User {user_id}:\n")
for movie, rating, genres in sorted_recommended_movies[:20]:
    print(f"{movie} - Predicted Rating: {rating:.2f}")

print("\nGeneralized Genre Recommendations:")
print(', '.join(list(common_genres_counter.keys())[:5]))