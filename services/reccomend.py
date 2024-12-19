import pandas as pd

def reccomend_books(books_title , books_df , cosine_sim):
    try:
        idx = books_df[books_df['Title'].str.lower() == books_title.lower()].index[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        sim_scores = sorted(sim_scores , key = lambda x : x[1], reverse=True)
        
        sim_scores = sim_scores[1:6]
        
        book_indices = [i[0] for i in sim_scores]
        
        return books_df['Title'].iloc[book_indices].tolist()
    except IndexError:
        return f"Book titled '{books_title}' not found in the dataset"