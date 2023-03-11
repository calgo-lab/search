import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import util
from utils import (bm25_tokenizer, get_bm25, get_data, get_sbert_embeddings,
                   load_biencoder, load_crossencoder)


# Define the search functions
# BM25 search (lexical search)= 100,
def search_candidate(
    query,
    n_candidate,
):
    bm25 = get_bm25()
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -n_candidate)[-n_candidate:]
    return [{"corpus_id": idx, "score": bm25_scores[idx]} for idx in top_n]


# Semantic Search
def search_biencoder(
    query,
    n_results,
):
    corpus_embeddings = get_sbert_embeddings()
    bi_encoder = load_biencoder()
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n_results)
    return hits[0]


# Re-Ranking
def re_ranking(
    query,
    bm25_hits,
    products,
):
    products["attributes_concat"] = products["attributes_concat"].astype(str)
    corpus_idx = [hit["corpus_id"] for hit in bm25_hits]
    cross_inp = [[query, products.loc[idx, "attributes_concat"]] for idx in corpus_idx]
    cross_encoder = load_crossencoder()
    cross_scores = cross_encoder.predict(cross_inp)
    min_score = min(cross_scores)
    max_score = max(cross_scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in cross_scores]
    return [
        {"corpus_id": hit["corpus_id"], "score": normalized_scores[idx]}
        for idx, hit in enumerate(bm25_hits)
    ]


def create_results_dataframe(
    hits, n_results, products_short, products_credibility, filter_credible_products
):
    if products_credibility != "all available products":
        hits = [hit for hit in hits if hit["corpus_id"] not in filter_credible_products]
    hits_df = pd.DataFrame.from_dict(
        sorted(hits, key=lambda x: x["score"], reverse=True)
    )[:n_results]
    return (
        pd.merge(
            hits_df, products_short, left_on="corpus_id", right_on=products_short.index
        )
        .round(2)
        .drop(columns="corpus_id")
    )


def retrieve_results(query, n_candidates, n_results, products_credibility):
    products, products_short, filter_credible_products = get_data()

    methods = [
        ("Lexical search (BM25)", search_candidate, n_candidates),
        ("Bi-Encoder Retrieval", search_biencoder, n_results),
        ("Cross-Encoder Re-ranked", re_ranking),
    ]

    for method_name, method_func, *args in methods:
        st.write(f"Top {n_results} {method_name} hits")
        with st.spinner(text=f"Loading {method_name} results..."):
            if method_name == "Lexical search (BM25)":
                candidates = method_func(query, args[0])
            if method_name == "Cross-Encoder Re-ranked":
                hits = method_func(query, candidates, products)
            else:
                hits = method_func(query, args[0])
            results_df = create_results_dataframe(
                hits, n_results, products_short, products_credibility, filter_credible_products
            )
        st.dataframe(results_df)
        st.write("\n-------------------------\n")


def main():
    # Set up the Streamlit app interface
    st.title("🔍 Product Search")
    with st.sidebar:
        if "n_candidates" not in st.session_state:
            st.session_state.n_candidates = 50
        if "n_results" not in st.session_state:
            st.session_state.n_results = 10
        if "products_credibility" not in st.session_state:
            st.session_state.products_credibility = "all available products"
        st.header("Filters:")
        products_credibility = st.radio(
            "Product kind",
            options=[
                "all available " "products",
                "only products with " "credible labels",
            ],
            index=0,
        )
        st.session_state.products_credibility = products_credibility
        n_candidates = st.radio("Number of candidates", options=[25, 50, 100], index=1)
        st.session_state.n_candidates = n_candidates
        n_results = st.radio(
            "Number of retrieved results", options=[5, 10, 15], index=1
        )
        st.caption(
            "BM25 retrieve n candidates and then a cross-encoder re-rank those candidates. "
            "All methods will retrieve selected number of results."
        )
        st.session_state.n_results = n_results

    if "query" not in st.session_state:
        st.session_state.query = ""
    query = st.text_input(
        "Enter a product search query"
    )
    search_button = st.button("Search")
    if st.session_state.query != query or search_button:
        if not query.strip():
            st.error("Query is empty", icon="🚨")
        else:
            st.session_state.query = query
            retrieve_results(
                st.session_state.query,
                st.session_state.n_candidates,
                st.session_state.n_results,
                st.session_state.products_credibility,
            )

            if (
                n_candidates != st.session_state.n_candidates
                or n_results != st.session_state.n_results
                or products_credibility != st.session_state.products_credibility
            ):
                retrieve_results(
                    st.session_state.query,
                    st.session_state.n_candidates,
                    st.session_state.n_results,
                    st.session_state.products_credibility,
                )


main()
