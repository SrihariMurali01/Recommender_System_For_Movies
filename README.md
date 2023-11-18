# Collaborative Filtering Recommender System with Neural Networks

## Overview

This project implements a collaborative filtering recommender system using neural networks. Collaborative filtering is a method to make automatic predictions about the preference of a user by collecting preferences from many users (collaborating). In this implementation, neural networks are utilized to learn latent representations of users and movies, enabling the model to make predictions for unseen user-movie pairs.

### DataSet
> The dataset is imported from Kaggle, and can be downloaded from [here](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system/download?datasetVersionNumber=1).

## Concepts Involved

### Collaborative Filtering

Collaborative filtering is a technique that makes automatic predictions about the interests of a user by collecting preferences from many users. It assumes that if a user A has the same opinion as a user B on an issue, A is more likely to have B's opinion on a different issue.

### Neural Networks

Neural networks are computational models inspired by the way biological neural networks in the human brain work. In this context, a neural network is used to learn complex patterns and representations from the interactions between users and movies.

### Embeddings

Embeddings are a crucial part of collaborative filtering models. They are low-dimensional, continuous representations of discrete entities like users and movies. Embeddings capture relationships and similarities between users and items, allowing the model to generalize well to unseen data.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Flask

Install dependencies using:

```bash
pip install pandas numpy scikit-learn tensorflow Flask
```

## Usage
1. **Data Preparation:** Ensure you have the movie and rating datasets. The movie dataset should contain 'movieId', 'genres', and other relevant columns. The rating dataset should have 'userId', 'movieId', and 'rating' columns.

2. **Run the Code:** Execute the Python script to train the collaborative filtering model and launch the Flask-based web frontend.

```bash
python collaborative_filtering.py
```

### Results
The model will be trained, and predictions will be made on the test set. The training progress and evaluation results will be displayed in the console. Additionally, a Flask-based web frontend has been provided for a more interactive user experience.

To interact with the recommender system, open your web browser and enter the URL displayed in the console.

## Fronted Application

For easy interaction and movie recommendations, you can access the Flask-based web frontend.
The implementation is provided here: [Frontend](/Frontend/)

You can run the model byt running the following command after downloading the dataset and the above frontend implementation source code

```bash
python app.py
```

### Acknowledgments
1. TensorFlow
2. Scikit-learn
3. Pandas
4. NumPy
5. Flask