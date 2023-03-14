import os
import pickle
import string
import zipfile
import streamlit as st

from sentence_transformers import CrossEncoder
from sklearn.feature_extraction import _stop_words

#dir_path = "/usr/src/app/models/data"
dir_path = "/Users/adrsanchez/PycharmProjects/demo_search/data"
# Tokenizer helper for BM25
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


# Load the pickled data
@st.cache_resource(show_spinner="Fetching GreenDB...")
def get_data():
    with zipfile.ZipFile(os.path.join(dir_path, "greedb_short.p.zip"), "r") as myzip:
        with myzip.open("greedb_short.p", "r") as f:
            products = pickle.load(f)
    products_short = products[
        ["name", "categories", "brand", "sustainability_labels", "colors", "url"]
    ]
    mask = ~products["sustainability_labels"].apply(
        lambda x: "certificate:OTHER" not in x and "certificate:UNKNOWN" not in x
    )
    filter_credible_products = products.index[mask]
    return (products, products_short, filter_credible_products)


# Load the pickled bm25 index
@st.cache_resource(show_spinner="Fetching BM25 corpus...")
def get_bm25():
    with zipfile.ZipFile(
        os.path.join(dir_path, "bm25_corpus_embeddings.p.zip"), "r"
    ) as myzip:
        with myzip.open("bm25_corpus_embeddings.p", "r") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner="Load cross-encoder model...")
def load_crossencoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        #CrossEncoder("/usr/src/app/models/stsb-distilroberta-base_plus_easy")


