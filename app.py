
import pickle
import streamlit as st
import numpy as np

# Define a function to compute cosine similarity
def compute_cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    dot_prod = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_prod / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

# Load pre-saved embedding dictionaries
def load_embedding_files():
    """Load embeddings for different models from pickle files."""
    positive_path = 'embed_skipgram_pos.pkl'
    negative_path = 'embed_skipgram_neg.pkl'
    glove_path = 'embed_glove.pkl'

    with open(positive_path, 'rb') as pos_file:
        pos_embeddings = pickle.load(pos_file)
    with open(negative_path, 'rb') as neg_file:
        neg_embeddings = pickle.load(neg_file)
    with open(glove_path, 'rb') as glove_file:
        glove_embeddings = pickle.load(glove_file)

    return pos_embeddings, neg_embeddings, glove_embeddings

# Find the most similar words for a given word
def find_similar_words(target_word, embedding_dict, top_n=10):
    """
    Identify top N words with the highest cosine similarity to the target word.

    Parameters:
    - target_word: Word for which similar words are sought.
    - embedding_dict: Dictionary containing word embeddings.
    - top_n: Number of similar words to retrieve.

    Returns:
    - List of the most similar words.
    """
    if target_word not in embedding_dict:
        return ["Word not in Corpus"]

    target_vec = embedding_dict[target_word]
    similarities = []

    for word, vec in embedding_dict.items():
        if word != target_word:  # Exclude the target word itself
            similarity = compute_cosine_similarity(target_vec, vec)
            similarities.append((word, similarity))

    # Rank words by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    return [word for word, _ in similarities[:top_n]]

# Main Streamlit app functionality
def main():
    # Load embedding dictionaries for all models
    pos_embeddings, neg_embeddings, glove_embeddings = load_embedding_files()

    # Display the app title and description
    st.title("Word Similarity Finder")
    st.write("Enter a word and choose a model to find similar words based on cosine similarity.")

    # Input field for user to type a word
    input_word = st.text_input("Type a word:", "example")  # Default word is "example"

    # Dropdown menu to select the embedding model
    selected_model = st.selectbox(
        "Select Embedding Model",
        ["GloVe Embeddings", "Skipgram Positive Embeddings", "Skipgram Negative Embeddings"]
    )

    # Select the appropriate embedding dictionary
    if selected_model == "GloVe Embeddings":
        embeddings = glove_embeddings
    elif selected_model == "Skipgram Positive Embeddings":
        embeddings = pos_embeddings
    elif selected_model == "Skipgram Negative Embeddings":
        embeddings = neg_embeddings

    # Display the top 10 similar words if input is provided
    if input_word:
        with st.spinner('Processing your request...'):
            similar_words = find_similar_words(input_word, embeddings, top_n=10)

            # Show results
            if similar_words == ["Word not in Corpus"]:
                st.error("The word you entered is not in the corpus.")
            else:
                st.success(f"Top 10 similar words for '{input_word}':")
                st.write(similar_words)

if __name__ == "__main__":
    main()
