import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Load precomputed embeddings
with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Convert embeddings to float32
embeddings = embeddings.astype("float32")

# Set up Faiss index
embedding_size = embeddings.shape[1]
n_clusters = 1
num_results = 5
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(
    quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT,
)
index.train(embeddings)
index.add(embeddings)

# Define search function
def search(query):
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding.reshape(1, -1)
    _, indices = index.search(query_embedding, num_results)
    images = [f"./images/image-{i+1}.jpg" for i in indices[0]]
    return images

# Streamlit UI
st.title("Image Search Engine")
query = st.text_input("Enter search query:")
if st.button("Search"):
    if query:
        results = search(query)
        st.image(results, width=200)
    else:
        st.warning("Please enter a search query.")
