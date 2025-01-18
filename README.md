# NLP_Assignment_1

This repository contains implementations of Word2Vec (Skip-gram, Skip-gram Negative Sampling) and GloVe, along with a Streamlit-based web application to search for the top 10 most similar words based on embeddings.

Files
Code and Models
A1_st124872.ipynb: Jupyter notebook with the implementation of Word2Vec and GloVe models.
app.py: Streamlit web application for word similarity search.
embed_glove.pkl: GloVe embeddings.
embed_skipgram_pos.pkl: Skip-gram positive sampling embeddings.
embed_skipgram_neg.pkl: Skip-gram negative sampling embeddings.
word-test-cleaned.txt: Cleaned word analogy dataset.
text_past_tense.txt: Extracted past-tense data from the word analogy dataset.
capital-common-countries.txt: Extracted capital-common-countries data from the word analogy dataset.
word-test.v1.txt: Original word analogy dataset.
wordsim_similarity_goldstandard.txt: Similarity dataset for Spearman correlation.

Run the Web App:
streamlit run app.py

Test Embeddings:
Query word embeddings and explore cosine similarity between words.
Analyze syntactic and semantic relationships using pre-trained embeddings.
