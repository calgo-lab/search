import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import util
from utils import bm25_tokenizer, get_data, get_bm25, get_sbert_embeddings, load_biencoder, load_crossencoder

# Define the search function
def search_candidate(query, products_short, n_candidate = 100, n_results = 15):
    n_candidate = n_candidate
    n_results = n_results
    products_short = products_short
    # BM25 search (lexical search)
    bm25 = get_bm25()
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -n_candidate)[-n_candidate:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits_df = pd.DataFrame.from_dict(sorted(bm25_hits, key=lambda x: x['score'],
                                                 reverse=True)[: n_results])
    results_bm25 = pd.merge(bm25_hits_df, products_short, left_on='corpus_id',
                            right_on=products_short.index).round(2)
    return (bm25_hits, results_bm25)


def search_biencoder(query, products_short, n_results = 15):
    products_short = products_short
    # Semantic Search
    corpus_embeddings = get_sbert_embeddings()
    bi_encoder = load_biencoder()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n_results)
    hits = pd.DataFrame.from_dict(sorted(hits[0], key=lambda x: x['score'], reverse=True))
    return pd.merge(hits, products_short, left_on='corpus_id', right_on=products_short.index).round(2)


def re_ranking(query, bm25_hits, products, products_short, n_results = 15):
    products = products
    products_short = products_short
    # Re-Ranking
    products['attributes_concat'] = products['attributes_concat'].astype(str)
    corpus_idx = [hit['corpus_id'] for hit in bm25_hits]
    cross_inp = [[query, products.loc[idx, 'attributes_concat']] for idx in corpus_idx]
    cross_encoder = load_crossencoder()
    cross_scores = cross_encoder.predict(cross_inp)
    re_ranked_hits = [{'corpus_id': hit['corpus_id'], 'cross_scores': cross_scores[idx]} for idx, hit in enumerate(bm25_hits)]
    re_ranked_hits_df = pd.DataFrame.from_dict(sorted(re_ranked_hits, key=lambda x: x['cross_scores'], reverse=True)[:n_results])
    return pd.merge(re_ranked_hits_df, products_short, left_on='corpus_id', right_on=products.index).round(2)



def main():
    # Set up the Streamlit app interface
    st.title("Product Search")
    query = st.text_input("Enter a product search query")
    products, products_short = get_data()
    if st.button("Search"):
        bm25_hits, results_bm25 = search_candidate(query, products_short)
        st.write("Top-15 lexical search (BM25) hits")
        st.dataframe(results_bm25)

        st.write("\n-------------------------\n")
        st.write("Top-15 Bi-Encoder Retrieval hits")
        st.dataframe(search_biencoder(query, products_short))

        st.write("\n-------------------------\n")
        st.write("Top-15 Cross-Encoder Re-ranker hits")
        st.dataframe(re_ranking(query, bm25_hits, products, products_short))
