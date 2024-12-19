# recommendation_engine.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
def load_data(file_path):
    book_data = pd.read_csv(file_path)
    
    # Create a combined "content" field by concatenating Title, Main Genre, and Sub Genre
    book_data['content'] = (
        book_data['Title'].fillna('') + ' ' + 
        book_data['Main Genre'].fillna('') + ' ' + 
        book_data['Sub Genre'].fillna('')
    )
    
    return book_data

# Vectorize the combined "content" column
def create_tfidf_matrix(book_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(book_data['content'])
    return tfidf_matrix, vectorizer

# Calculate similarity matrix
def calculate_similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Get recommendations based on title
def get_recommendations(title, book_data, cosine_sim, top_n=5):
    try:
        idx = book_data[book_data['Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # Exclude the book itself
        book_indices = [i[0] for i in sim_scores]
        return book_data['Title'].iloc[book_indices].tolist()
    except IndexError:
        return ["Book not found"]

# Load data and create matrix on module load
book_data = load_data('E:\Web projects\BookReccomendationSystem\services\Books_df.csv')  # Replace with your file path
tfidf_matrix, vectorizer = create_tfidf_matrix(book_data)
cosine_sim = calculate_similarity_matrix(tfidf_matrix)
