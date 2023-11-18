import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder  # Label Encoding for User and Movie Ids

from keras.models import load_model


df_movies = pd.read_csv('movies.csv').head(500000)
df_ratings = pd.read_csv('ratings.csv').head(500000)

# Merging movies and their ratings given
dfMerged = pd.merge(df_ratings, df_movies, on='movieId')

userEncoder = LabelEncoder()
movieEncoder = LabelEncoder()

dfMerged['userEncoded'] = userEncoder.fit_transform(dfMerged['userId'])
dfMerged['movieEncoded'] = movieEncoder.fit_transform(dfMerged['movieId'])


# Acquire the unique users and movies
Vu = dfMerged['userEncoded'].nunique()
Vm = dfMerged['movieEncoded'].nunique()

model = load_model('best_model.h5')


user_id = 1
movie_id = 307

# Encode user and movie IDs
user_encoded = userEncoder.transform([user_id])[0]
movie_encoded = movieEncoder.transform([movie_id])[0]

# Make predictions using the trained model
prediction = model.predict([np.array([user_encoded]), np.array([movie_encoded])])[0][0]
movieTitle = df_movies.loc[df_movies['movieId'] == movie_id, 'title'].values[0]

print(f"Predicted rating for user {user_id} and movie \"{movieTitle}\": {prediction: 0.3f}", end=" ")
for i in range(int(prediction)):
    print("⭐", end=" ")


