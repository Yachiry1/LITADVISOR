import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('GoodReads_100k_books.csv')
    df = df.dropna(subset=['desc', 'genre', 'title', 'author'])

    # TF-IDF for description
    tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
    X_desc = tfidf_desc.fit_transform(df['desc'])

    # Dimensionality reduction for description
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_desc_svd = svd.fit_transform(X_desc)

    # Genre list processing
    df['genre_list'] = df['genre'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer(sparse_output=True)
    X_genre = mlb.fit_transform(df['genre_list'])

    # Numeric features
    numeric_features = ['pages', 'rating', 'reviews', 'totalratings']
    for col in numeric_features:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[numeric_features])

    # TF-IDF for title and author
    tfidf_title = TfidfVectorizer(max_features=1000, stop_words='english')
    X_title = tfidf_title.fit_transform(df['title'])

    tfidf_author = TfidfVectorizer(max_features=1000, stop_words='english')
    X_author = tfidf_author.fit_transform(df['author'])

    # Combine features
    X_combined = hstack([X_desc_svd, X_genre, X_title, X_numeric, X_author])
    X_combined = normalize(X_combined, axis=1)

    return df.reset_index(drop=True), X_combined, tfidf_title, tfidf_author, mlb, tfidf_desc, svd, scaler


df, X_combined, tfidf_title, tfidf_author, mlb, tfidf_desc, svd, scaler = load_and_process_data()

@st.cache_resource
def get_knn_model(_X):
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(_X)
    return model

knn_model = get_knn_model(X_combined)

# Get recommendations by existing title
def get_recommendations_by_title(title, df, knn_model, X_combined):
    idx_list = df[df['title'].str.lower() == title.lower()].index.tolist()
    if not idx_list:
        return None
    idx = idx_list[0]
    distances, indices = knn_model.kneighbors(X_combined[idx], return_distance=True)
    similar_indices = indices.flatten()[1:]
    return df.iloc[similar_indices][['title', 'author', 'genre', 'desc', 'pages', 'rating', 'reviews', 'totalratings']]

# Get recommendations by user input
def get_recommendations_by_input(title, genre, author, desc, pages, rating, reviews, totalratings,
                                  tfidf_title, tfidf_author, tfidf_desc, svd, mlb, scaler, knn_model, X_combined, df):
    desc_tfidf = tfidf_desc.transform([desc])
    desc_vec = svd.transform(desc_tfidf)
    title_vec = tfidf_title.transform([title])
    author_vec = tfidf_author.transform([author])
    genre_list = genre.split(',')
    genre_vec = mlb.transform([genre_list])
    num_input = scaler.transform([[pages, rating, reviews, totalratings]])
    combined = hstack([desc_vec, genre_vec, title_vec, num_input, author_vec])
    combined = normalize(combined)
    distances, indices = knn_model.kneighbors(combined, return_distance=True)
    similar_indices = indices.flatten()
    return df.iloc[similar_indices][['title', 'author', 'genre', 'desc', 'pages', 'rating', 'reviews', 'totalratings']]

# Streamlit UI
st.title("📚 LITADVISOR")
tab1, tab2, tab3 = st.tabs(["📊 View Dataset", "🔍 Recommend by Title", "📝 Recommend by New Book"])

with tab1:
    st.subheader("Full Dataset")
    st.dataframe(df[['title', 'author', 'genre', 'desc', 'pages', 'rating', 'reviews', 'totalratings']])

with tab2:
    st.subheader("Get Recommendations by Book Title")
    book_title = st.text_input("Enter book title:")
    if st.button("Recommend (existing book)"):
        recs = get_recommendations_by_title(book_title, df, knn_model, X_combined)
        if recs is not None:
            st.write("Top 5 Recommendations:")
            st.dataframe(recs)
        else:
            st.warning("Book not found in dataset.")

with tab3:
    st.subheader("Get Recommendations by New Book Details")
    new_title = st.text_input("Title:")
    new_genre = st.text_input("Genre (comma-separated):")
    new_author = st.text_input("Author:")
    new_desc = st.text_area("Description:")
    new_pages = st.number_input("Pages:", min_value=1, max_value=5000, value=200)
    new_rating = st.number_input("Rating:", min_value=0.0, max_value=5.0, value=4.0)
    new_reviews = st.number_input("Reviews:", min_value=0, value=100)
    new_totalratings = st.number_input("Total Ratings:", min_value=0, value=500)

    if st.button("Recommend (new book)"):
        if new_title and new_genre and new_author and new_desc:
            recs = get_recommendations_by_input(
                new_title, new_genre, new_author, new_desc,
                new_pages, new_rating, new_reviews, new_totalratings,
                tfidf_title, tfidf_author, tfidf_desc, svd, mlb, scaler,
                knn_model, X_combined, df
        )
            st.write("Top 5 Recommendations:")
            st.dataframe(recs)
        else:
            st.warning("Please fill all fields.")
