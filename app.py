import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time  # For retry mechanism

# TMDB API Key
TMDB_API_KEY = '7f253a494fa13a3d61cc8b02669b09b5'
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Image"

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert2(obj):
        return [i['name'] for i in ast.literal_eval(obj)[:3]]  # Top 3 actors

    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert2)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    df = movies[['movie_id', 'title', 'tags']]
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

    ps = PorterStemmer()
    df['tags'] = df['tags'].apply(lambda x: " ".join(ps.stem(i) for i in x.split()))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return df, similarity

# Improved Movie Poster Fetching with Retry & Fallback
def get_movie_poster(movie_title, retries=3, delay=2):
    for attempt in range(retries):
        try:
            search_url = f'https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}'
            response = requests.get(search_url, timeout=5).json()

            if response.get('results'):
                poster_path = response['results'][0].get('poster_path')
                if poster_path:
                    return f'https://image.tmdb.org/t/p/w500{poster_path}'
            return PLACEHOLDER_IMAGE  # If no poster is found
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)  # Retry after delay
            else:
                st.warning(f"Could not fetch poster for {movie_title} due to network error.")
                return PLACEHOLDER_IMAGE  # Fallback image

# Movie Recommendation Function
def recommend(movie, df, similarity):
    try:
        movie_index = df[df['title'] == movie].index[0]
        distance = similarity[movie_index]
        movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
        return [df.iloc[i[0]].title for i in movies_list]
    except IndexError:
        return []

# Main Streamlit App
def main():
    st.set_page_config(page_title='Movie Recommender', page_icon='🎬', initial_sidebar_state='collapsed')

    # Dark Mode Toggle
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    col1, col2 = st.columns([9, 1])
    with col2:
        theme_toggle = st.toggle('Dark Mode', value=st.session_state.theme == 'dark')

    # Improved Dark Mode Styling
    if theme_toggle:
        st.session_state.theme = 'dark'
        dark_css = """
        <style>
        .stApp { background-color: #121212 !important; color: white !important; }
        .stButton>button { background-color: #bb86fc !important; color: white !important; border-radius: 10px; }
        .stSelectbox div { background-color: #1f1f1f !important; color: white !important; }
        .stColumns div { background-color: #1f1f1f !important; padding: 10px; border-radius: 10px; }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        st.session_state.theme = 'light'
        light_css = """
        <style>
        .stApp { background-color: white !important; color: black !important; }
        .stButton>button { background-color: #ff4b4b !important; color: white !important; border-radius: 10px; }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

    # Title
    st.title('🎬 Movie Recommender System')

    # Load data
    df, similarity = load_and_preprocess_data()

    # Movie Selection
    selected_movie = st.selectbox(
        'Select a movie:',
        sorted(df['title'].unique())
    )

    # Recommendation Button
    if st.button('Get Recommendations'):
        recommendations = recommend(selected_movie, df, similarity)

        # Display Recommendations
        if recommendations:
            st.subheader(f'Movies Similar to {selected_movie}:')

            cols = st.columns(5)
            for i, movie in enumerate(recommendations):
                with cols[i]:
                    poster_url = get_movie_poster(movie)
                    st.image(poster_url, use_container_width=True)
                    st.write(movie)
        else:
            st.warning('No recommendations found.')

if __name__ == '__main__':
    main()


