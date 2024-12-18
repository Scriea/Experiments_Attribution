import pickle
import os
import json
import argparse

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--gfile", help="Path to grounded JSON file", required=True, type=str)
parser.add_argument("--rfile", help="Path to response pickle file", required=True, type=str)
parser.add_argument("--tss-index", help="Path to TSS selected passage's indexes", required=True, type=str)
parser.add_argument("--output-dir", help="Output directory name", required=True, type=str)
args = parser.parse_args()

# Input paths
path = args.gfile
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Load response file
with open(args.rfile, 'rb') as f:
    response = pickle.load(f)

# Load index file
with open(args.tss_index, 'r') as f:
    index = json.load(f)

# Load grounding file
with open(path, 'r') as f:
    grounding_file = json.load(f)

# Initialize data structure for outputs
autoais_formats = [{}, {}, {}]  # One dictionary for each file

# Process each question
for question_id, item in enumerate(grounding_file):
    question = item['question']
    selected_indices = index[str(question_id)][0]  # Get the selected indices for the question

    for i, context_idx in enumerate(selected_indices):
        if i < 3:  # Ensure we only process up to 3 contexts
            context_text = item['ctxs'][context_idx]['text']
            autoais_formats[i][question] = {
                'question': question,
                'answer': response[question]['answer'],
                'passage': context_text,
                'attribution': ''
            }

identity = os.path.basename(args.tss_index).split('.')[0]
dirname = os.path.basename(os.path.dirname(os.path.abspath(args.tss_index)))

# Save the output files
for i in range(3):
    output_file = os.path.join(output_dir, f"{dirname}_{identity}_{i + 1}.pkl")
    print(output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(autoais_formats[i], f)

print(f"Saved 3 output files in {output_dir}")
