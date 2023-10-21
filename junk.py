import pickle

f = pickle.load(open("data/finetune_data_600_plus_url.pkl", "rb"))

print(f[1])
exit()
import numpy as np
from apricot import FeatureBasedSelection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (replace this with your actual dataset)
dataset = [
    {
        "question": "What is the capital of France?",
        "context": "Paris is the capital of France.",
        "answer": "Paris",
    },
    {
        "question": "Who wrote the book 'To Kill a Mockingbird'?",
        "context": "Harper Lee wrote the book 'To Kill a Mockingbird'.",
        "answer": "Harper Lee",
    },
    # Add more entries here
]

# Query question
q = "What is the capital of Italy?"

# Number of samples to select
k = 3

# Extract features from the dataset using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([d["question"] for d in dataset])

# Calculate cosine similarities between the query and dataset questions
query_vector = vectorizer.transform([q])
similarities = cosine_similarity(query_vector, X)

# Flatten the similarities matrix
similarities = similarities.flatten()

# Initialize the FeatureBasedSelection object
model = FeatureBasedSelection(similarities)

# Select the top-k samples
selected_indices = model.fit_transform(X, k)

# Get the selected samples from the dataset
selected_samples = [dataset[i] for i in selected_indices]

# Print the selected samples
for sample in selected_samples:
    print(sample)
