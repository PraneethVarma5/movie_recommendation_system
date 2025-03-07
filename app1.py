import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit App Title
st.title("🎬 Movie Recommender System")

# File Upload
uploaded_file = st.file_uploader("Upload movies.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display dataset info
    st.write(f"📊 Data Loaded - DF Shape: {df.shape}")

    # Ensure 'title' and 'overview' columns exist
    if "title" not in df.columns or "overview" not in df.columns:
        st.error("Error: Dataset must contain 'title' and 'overview' columns.")
    else:
        # Fill missing overviews with empty string
        df["overview"] = df["overview"].fillna("")

        # Convert text data to TF-IDF features
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df["overview"])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Dropdown to select a movie
        selected_movie = st.selectbox("🎥 Select a movie:", df["title"].values)

        # Function to recommend movies
        def recommend(movie_name, df, similarity_matrix):
            if movie_name not in df["title"].values:
                return ["Movie not found in dataset"]
            
            # Get index of the selected movie
            movie_idx = df[df["title"] == movie_name].index[0]

            # Get similarity scores and sort
            similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5

            # Fetch recommended movie names
            recommended_movies = [df.iloc[i[0]]["title"] for i in similarity_scores]
            return recommended_movies

        # Button to get recommendations
        if st.button("🔍 Get Recommendations"):
            recommendations = recommend(selected_movie, df, similarity_matrix)

            if recommendations:
                st.success("✅ Recommended Movies:")
                for movie in recommendations:
                    st.write(f"- {movie}")
            else:
                st.warning("⚠️ No recommendations found.")
else:
    st.warning("⚠️ Please upload the movies.csv file.")
