"""
Movie Recommendation Skill.

- movies like <movie-name>

"""
import numpy as np
import pandas as pd
from nltk import edit_distance

# Local Imports.
from backend.config import cosine_sim_scores_path, movie_data_path

def find_nearest_title(user_input_title):
    """
    Checks for nearest movie title in dataset

    Parameters
    ----------
        user_input_title: str.

    Returns
    -------
        nearest_title
    """
    movies = pd.read_csv(movie_data_path)
    movie_titles = movies["title"]
    distances = {}

    for titles in movie_titles:
        distances[titles] = edit_distance(user_input_title, titles)

    sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=False)
    nearest_title = sorted_distances[0][0]
    return nearest_title


def get_movie_plot(user_input_tokens):
    """
    Returns movie's summary.

    Parameters
    ----------
        user_input_tokens: list.

    Returns
    -------
        summary.
    """
    # Process movie title from user.
    user_input_title = user_input_tokens[1:]
    user_input_title = ' '.join(user_input_title)

    # Find nearest title.
    movie_title = find_nearest_title(user_input_title)
    movie_data = pd.read_csv(movie_data_path)

    # Find Plot.
    plot = movie_data[movie_data["title"] == movie_title]["summary"].values[0]
    year_of_release = movie_data[movie_data["title"] == movie_title]["year_of_release"].values[0]
    genre = movie_data[movie_data["title"] == movie_title]["genres"].values[0]

    # Format Response.
    movie_plot = f"{movie_title.capitalize()} ({year_of_release}, {genre}): {plot}"
    return movie_plot


def get_recommendations(user_input_tokens):
    """
    Computes Top 5 movie recommendation.

    Parameters
    ----------
        user_input_tokens: tokenized input.

    Returns
    -------
        5 similar movies.
    """
    # Process movie title from user input.
    user_input_title = user_input_tokens[2:]
    user_input_title = ' '.join(user_input_title)
    movie_title = find_nearest_title(user_input_title)

    # Read files from db.
    movie_data = pd.read_csv(movie_data_path)
    cosine_sim_scores = np.loadtxt(cosine_sim_scores_path)

    # Construct titles dictionary.
    titles = pd.Series(movie_data.index, index=movie_data["title"])

    # idx for user input.
    input_title_idx = titles[movie_title]

    # compute cosine similarity.
    cosine_score = list(enumerate(cosine_sim_scores[input_title_idx]))
    cosine_score = sorted(cosine_score, key=lambda x: x[1], reverse=True)

    # idx of top similar movies.
    similar_movies_idx = [i[0] for i in cosine_score]
    similar_movies_idx = similar_movies_idx[1:6]

    # process recommendation as list.
    similar_movies = list(titles.iloc[similar_movies_idx].index)
    similar_movies = [movie.title() for movie in similar_movies]
    recommendations = f"Recommendations ({movie_title.title()}) --> {', '.join(similar_movies)}."
    return recommendations



