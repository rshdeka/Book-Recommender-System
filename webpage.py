import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Load the pickle files
books_dict = pickle.load(open('books_dict.pkl', 'rb'))
popular_df_dict = pickle.load(open('popular_df_dict.pkl', 'rb'))
ratings_matrix = pickle.load(open('ratings_matrix.pkl', 'rb'))
similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))

# Create a dataframe
books = pd.DataFrame(books_dict)
popular_df = pd.DataFrame(popular_df_dict)

html_temp = """ 
<div style = "background-color: #63e0c7; padding: 10px">
<h2 style = "color: white; text-align: center;">Book Recommender System
</div>
<div style = "background-color: white; padding: 5px">
<p style= "color: #099e80; text-align: center; font-family: Courier; font-size: 15px;">
<i>Not sure what to read next? Let's recommend you something...</i></p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

selected_book = st.selectbox("What are you looking for today?", popular_df['Book-Title'].values)


def recommend(book):
    # Get index of a book
    book_index = np.where(ratings_matrix.index==book)[0][0]
    # To get the distance with every other book, we need to get its index in the similarity matrix
    distances = similarity_matrix[book_index]
    # Sort the books acc. to similarity
    books_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])
    # Fetch the top 10 similar books
    recommended_books = []
    recommended_books_posters = []
    for i in books_list[1:11]:
        book_indices = i[0]
        temp_df = books[books['Book-Title'] == ratings_matrix.index[book_indices]]
        recommended_books.append(ratings_matrix.index[book_indices])
        recommended_books_posters.append(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
    return recommended_books, recommended_books_posters


if st.button('Show Recommendations'):
    names, posters = recommend(selected_book)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(names[0])
        image1 = Image.open(posters[0])
        st.image(image1)
    with col2:
        st.markdown(names[1])
        image2 = Image.open(posters[1])
        st.image(image2)
    with col3:
        st.markdown(names[2])
        image3 = Image.open(posters[2])
        st.image(image3)
    with col1:
        st.markdown(names[3])
        image4 = Image.open(posters[3])
        st.image(image4)
    with col2:
        st.markdown(names[4])
        image5 = Image.open(posters[4])
        st.image(image5)
    with col3:
        st.markdown(names[5])
        image6 = Image.open(posters[5])
        st.image(image6)
    with col1:
        st.markdown(names[6])
        image7 = Image.open(posters[6])
        st.image(image7)
    with col2:
        st.markdown(names[7])
        image8 = Image.open(posters[7])
        st.image(image8)
    with col3:
        st.markdown(names[8])
        image9 = Image.open(posters[8])
        st.image(image9)
    with col1:
        st.markdown(names[9])
        image10 = Image.open(posters[9])
        st.image(image10)


html_temp1 = """
    <div style = "background-color: #63e0c7">
    <p style = "color: white; text-align: center;">Designed & Developed By: <b>Rajashri Deka</b></p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)