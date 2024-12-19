from flask import Flask,render_template,request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from reccomend import reccomend_books

app = Flask(__name__)

def load_data():

    file_path = r"E:\Web projects\BookReccomendationSystem\services\Books_df.csv"
    books_df = pd.read_csv(file_path)

    # print(books_df.head())

    def clean_price(price):
        try:
            return float(str(price).replace('₹' , '').replace(',' , '').strip())
        except:
            return 0.0

    books_df['Price'] = books_df['Price'].apply(clean_price)

    books_df.fillna('',inplace=True)

    scaler = MinMaxScaler()
    books_df[['Price' , 'Rating' , 'No. of People rated']] = scaler.fit_transform(
        books_df[['Price' , 'Rating' , 'No. of People rated']]
    )

    books_df['combined_features'] = (
        books_df['Title'] + ' '+
        books_df['Author'] + ' '+
        books_df['Main Genre'] * 3 + ' ' +
        books_df['Sub Genre'] * 2 + ' '+ 
        books_df['Type']+' '+
        books_df['Price'].astype(str)+' '+
        books_df['Rating'].astype(str)+' '+
        books_df['No. of People rated'].astype(str)
        
    )

    # print(books_df['combined_features'].head())

    tfidf = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf.fit_transform(books_df['combined_features'])

    # print(f"TF-IDF Matrix Shape : {tfidf_matrix.shape}")

    cosine_sim = cosine_similarity(tfidf_matrix , tfidf_matrix)
    
    return books_df , cosine_sim

# print(f"Cosine Similarity Matrix Shape : {cosine_sim.shape}")

# book_title = "The 7 Habits of Highly Effective People: Infographics Edition: Powerful Lessons in Personal Change"
# reccmendations = reccomend_books(book_title , books_df , cosine_sim)
# print(f"Recommendations for '{book_title}' :")
# for i , rec in enumerate(reccmendations , start=1):
#     print(f"{i}.{rec}")

books_df , cosine_sim = load_data()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/recommend' , methods = ['POST'])
def recommend():
    book_title = request.form['book_title']
    if book_title.strip() == "":
        return jsonify({'error' : 'Please enter a valid book title'})
    else:
        recommendations = reccomend_books(book_title , books_df , cosine_sim)
        if isinstance(recommendations , str):
            return jsonify({'error' : recommendations})
        else:
            return jsonify({'recommendations' : recommendations})

if __name__ == '__main__':
    app.run(debug = True)
            
        

