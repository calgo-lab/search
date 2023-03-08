import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import util
from utils import (bm25_tokenizer, get_bm25, get_data, get_sbert_embeddings,
                   load_biencoder, load_crossencoder)


# Define the search functions
# BM25 search (lexical search)= 100,
def search_candidate(query, products_short, n_candidate, n_results):
    bm25 = get_bm25()
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -n_candidate)[-n_candidate:]
    bm25_hits = [{"corpus_id": idx, "score": bm25_scores[idx]} for idx in top_n]
    bm25_hits_df = pd.DataFrame.from_dict(
        sorted(bm25_hits, key=lambda x: x["score"], reverse=True)[:n_results]
    )
    results_bm25 = (
        pd.merge(
            bm25_hits_df,
            products_short,
            left_on="corpus_id",
            right_on=products_short.index,
        )
        .round(2)
        .drop(columns="corpus_id")
    )
    return (bm25_hits, results_bm25)


# Semantic Search
def search_biencoder(query, products_short, n_results):
    corpus_embeddings = get_sbert_embeddings()
    bi_encoder = load_biencoder()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n_results)
    hits = pd.DataFrame.from_dict(
        sorted(hits[0], key=lambda x: x["score"], reverse=True)
    )
    return (
        pd.merge(
            hits, products_short, left_on="corpus_id", right_on=products_short.index
        )
        .round(2)
        .drop(columns="corpus_id")
    )


# Re-Ranking
def re_ranking(query, bm25_hits, products, products_short, n_results):
    products["attributes_concat"] = products["attributes_concat"].astype(str)
    corpus_idx = [hit["corpus_id"] for hit in bm25_hits]
    cross_inp = [[query, products.loc[idx, "attributes_concat"]] for idx in corpus_idx]
    cross_encoder = load_crossencoder()
    cross_scores = cross_encoder.predict(cross_inp)
    re_ranked_hits = [
        {"corpus_id": hit["corpus_id"], "cross_scores": cross_scores[idx]}
        for idx, hit in enumerate(bm25_hits)
    ]
    re_ranked_hits_df = pd.DataFrame.from_dict(
        sorted(re_ranked_hits, key=lambda x: x["cross_scores"], reverse=True)[
            :n_results
        ]
    )
    return (
        pd.merge(
            re_ranked_hits_df,
            products_short,
            left_on="corpus_id",
            right_on=products.index,
        )
        .round(2)
        .drop(columns="corpus_id")
    )


def retrieve_results(query, n_candidates, n_results):
    products, products_short = get_data()

    st.write(f"Top {st.session_state.n_results} lexical search (BM25) hits")
    with st.spinner(text="Loading BM25 results..."):
        st.session_state.bm25_hits, st.session_state.results_bm25 = search_candidate(
            query, products_short, n_candidates, n_results
        )
    st.dataframe(st.session_state.results_bm25)

    st.write("\n-------------------------\n")
    st.write(f"Top {st.session_state.n_results} Bi-Encoder Retrieval hits")
    with st.spinner(text="Loading SBERT results..."):
        st.session_state.results_biencoder = search_biencoder(
            query, products_short, n_results
        )
    st.dataframe(st.session_state.results_biencoder)

    st.write("\n-------------------------\n")
    st.write(f"Top {st.session_state.n_results} Cross-Encoder Re-ranker hits")
    with st.spinner(
        text=f"Reranking top {st.session_state.n_candidates} BM25 candidates..."
    ):
        st.session_state.results_reranked = re_ranking(
            query, st.session_state.bm25_hits, products, products_short, n_results
        )
    st.dataframe(st.session_state.results_reranked)


def main():
    # Set up the Streamlit app interface
    st.title("Product Search")
    with st.sidebar:
        if "num_candidates" not in st.session_state:
            st.session_state.n_candidates = 50
        if "num_results" not in st.session_state:
            st.session_state.n_results = 10
        n_candidates = st.radio("Number of candidates", options=[25, 50, 100], index=1)
        st.session_state.n_candidates = n_candidates
        n_results = st.radio(
            "Number of retrieved results", options=[5, 10, 15], index=1
        )
        st.session_state.n_results = n_results

    if "query" not in st.session_state:
        st.session_state.query = ""
    query = st.text_input("Enter a product search query", value=st.session_state.query)
    search_button = st.button("Search")
    if search_button:
        st.session_state.query = query
        retrieve_results(
            st.session_state.query,
            st.session_state.n_candidates,
            st.session_state.n_results,
        )

        if n_candidates != st.session_state.n_candidates:
            retrieve_results(
                st.session_state.query,
                st.session_state.n_candidates,
                st.session_state.n_results,
            )

        if n_results != st.session_state.n_results:
            retrieve_results(
                st.session_state.query,
                st.session_state.n_candidates,
                st.session_state.n_results,
            )

main()
