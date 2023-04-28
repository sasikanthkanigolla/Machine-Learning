import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

data = pd.read_csv("C:\\Users\\Asus\\Desktop\\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

model = SentenceTransformer('msmarco-MiniLM-L-12-v3')
restaurant_reviews = data["Review"].tolist()

hotel_reviews_embds = model.encode(restaurant_reviews)


index = faiss.IndexFlatL2(hotel_reviews_embds.shape[1])
index.add(hotel_reviews_embds)
faiss.write_index(index, 'index_restaurant_reviews')
index = faiss.read_index('index_restaurant_reviews')


a = st.text_input("Enter the text to search")


submit_button = st.button("Submit")
if submit_button:
        query_vector = model.encode([a])
        k = 5
        top_k = index.search(query_vector, k)

        for _id in top_k[1].tolist()[0]:
            st.write(restaurant_reviews[_id])
