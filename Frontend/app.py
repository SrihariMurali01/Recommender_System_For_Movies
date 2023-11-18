from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tqdm import tqdm
import os
from collections import Counter
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.static_folder = 'static'

# Load data and model
df_movies = pd.read_csv('movies.csv').head(1000)
df_ratings = pd.read_csv('ratings.csv').head(1048576)

dfMerged = pd.merge(df_ratings, df_movies, on='movieId')

userEncoder = LabelEncoder()
movieEncoder = LabelEncoder()

dfMerged['userEncoded'] = userEncoder.fit_transform(dfMerged['userId'])
dfMerged['movieEncoded'] = movieEncoder.fit_transform(dfMerged['movieId'])

model = load_model('best_model.h5')

# Function to get movie recommendation
def get_movie_recommendations(user_id):
    recommended_movies = []

    for movie_id in tqdm(df_movies['movieId'], desc="Processing Movie Preferences"):
        user_encoded = userEncoder.transform([user_id])[0]
        try:
            movie_encoded = movieEncoder.transform([movie_id])[0]
        except ValueError:
            continue

        prediction = model.predict([np.array([user_encoded]), np.array([movie_encoded])], verbose=0)[0][0]
        movie_title = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values[0]
        movie_genres = df_movies.loc[df_movies['movieId'] == movie_id, 'genres'].values[0]

        if prediction >= 2.5:
            recommended_movies.append((movie_title, prediction, movie_genres.split('|')))

    # Aggregating the genres
    all_genres = [genre for _, _, genres in recommended_movies for genre in genres]
    common_genres_counter = Counter(all_genres)

    # Sort recommended movies by predicted ratings
    sorted_recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)

    return sorted_recommended_movies[:10], list(common_genres_counter.keys())[:5]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = int(request.form['user_id'])
    recommended_movies, generalized_genres = get_movie_recommendations(user_id)
    return render_template('recommendations.html', user_id=user_id, recommended_movies=recommended_movies, generalized_genres=generalized_genres)

if __name__ == '__main__':
    app.run(debug=True, host=socket.gethostbyname(socket.gethostname()))

