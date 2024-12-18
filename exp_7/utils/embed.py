import os
import argparse
import pickle
import json
from sentence_transformers import SentenceTransformer

def main(input_file, output_file):
    # Load input JSON file
    with open(input_file, 'r') as myfile:
        grounded_data = myfile.read()
    obj = json.loads(grounded_data)

    # Process data
    retrieved_contexts = {}
    for item in obj:
        query = item['question']
        retrieved_contexts[query] = []
        for context_index in range(100):
            retrieved_context = item['ctxs'][context_index]['text']
            retrieved_contexts[query].append(retrieved_context)

    # Flatten contexts
    retrieved_contexts_together = []
    for query, contexts in retrieved_contexts.items():
        retrieved_contexts_together += contexts

    # Load the model and encode contexts
    model = SentenceTransformer("gtr-t5-xxl")
    model.half()
    print("Model Loaded")

    embeddings = model.encode(retrieved_contexts_together, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_file.endswith('.pkl'):
        # Save embeddings to output file
        with open(output_file, 'wb') as handle:
            pickle.dump(embeddings, handle)
        print(f"Embeddings saved to {output_file}")
    else:
        print("Output file must be a pickle file.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and generate embeddings for retrieved contexts.")
    parser.add_argument("--gfile", type=str, help="Path to the input JSON file.")
    parser.add_argument("--outfile", type=str, help="Path to save the output pickle file.")
    
    args = parser.parse_args()

    main(args.gfile, args.outfile)
