import numpy as np
import pandas as pd
import streamlit as st
import pickle
import base64
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

# Set background image
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
# Set background from local folder
set_background('judith-frietsch-IHZYwmTGnUc-unsplash.jpg')


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
    for i in books_list[1:11]:
        book_indices = i[0]
        temp_df = books[books['Book-Title'] == ratings_matrix.index[book_indices]]
        recommended_books.append(ratings_matrix.index[book_indices])
    return recommended_books


if st.button('Show Recommendations'):
    recommendations = recommend(selected_book)
    for i in recommendations:
        st.markdown(i)


html_temp1 = """
    <div style = "background-color: #63e0c7">
    <p style = "color: white; text-align: center;">Designed & Developed By: <b>Rajashri Deka</b></p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)