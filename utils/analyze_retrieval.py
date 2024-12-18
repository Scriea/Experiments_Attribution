
# import argparse
# import json
# import pickle
# from collections import Counter
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch

# # Load AutoAIS model and tokenizer
# AUTOAIS = "google/t5_xxl_true_nli_mixture"
# hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
# hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS, device_map="auto")

# def format_example_for_autoais(question, passage, answer):
#     """Format example for AutoAIS evaluation."""
#     input_text = f"premise: {passage} hypothesis: The answer to the question '{question}' is '{answer}'"
#     return input_text

# def infer_autoais(question, passage, answer):
#     """Run AutoAIS inference."""
#     input_text = format_example_for_autoais(question, passage, answer)
#     input_ids = hf_tokenizer(input_text, return_tensors="pt").to("cuda").input_ids
#     outputs = hf_model.generate(input_ids)
#     result = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return "Y" if result == "1" else "N"

# # Load the JSON files
# def load_json(file_path):
#     with open(file_path, 'r') as f:
#         return json.load(f)

# # Load inference results from .pkl file
# def load_inference_results(file_path):
#     with open(file_path, 'rb') as f:
#         return pickle.load(f)

# # Extract question and top N retrieved passages for a given question ID
# def extract_question_and_contexts(data, question_id, top_n=5):
#     for item in data:
#         if item['id'] == question_id:
#             question = item['question']
#             contexts = item['ctxs'][:top_n]
#             return question, contexts
#     return None, []

# # Compare retrievals across settings
# def compare_retrievals(setting1, setting2, setting3_files, inference_results, question_id, top_n=5):
#     # Load data for each setting
#     data1 = load_json(setting1)
#     data2 = load_json(setting2)
#     data3_list = [load_json(file) for file in setting3_files]
    

#     # Extract question and top contexts
#     question, top_contexts_1 = extract_question_and_contexts(data1, question_id, top_n)
#     _, top_contexts_2 = extract_question_and_contexts(data2, question_id, top_n)
#     top_contexts_3_combined = []

#     answers = inference_results[question]['answer']
#     for data3 in data3_list:
#         _, step_contexts = extract_question_and_contexts(data3, question_id, top_n)
#         top_contexts_3_combined.extend(step_contexts)

#     top_contexts_3_combined = top_contexts_3_combined[:top_n]
#     print("##########")
#     # print(top_contexts_3_combined)
#     # Analyze and display results
#     print(f"\nQuestion (ID {question_id}): {question}\n Asnwer: {answers}")



#     def evaluate_contexts(contexts, setting_name):
#         print(f"{setting_name}:")
#         for i, ctx in enumerate(contexts, 1):
#             passage = ctx['text']
#             answer = answers  # Assuming the first answer is the reference
#             autoais_result = infer_autoais(question, passage, answer)
#             print(f"{i}. {passage}\n   AutoAIS Evaluation: {autoais_result}\n")

#     def evaluate_diverse_contexts(setting3_files, inference_results, question_id):
#         # Load data for each setting in Setting 3
#         diverse_passages = []
#         for file in setting3_files:
#             data = load_json(file)
#             _, contexts = extract_question_and_contexts(data, question_id, top_n=1)
#             if contexts:
#                 diverse_passages.append(contexts[0]['text'])

#         # Combine top passages from each file
#         combined_passage = " ".join(diverse_passages)
#         question = next(iter(inference_results))[question_id]["question"]
#         answer = inference_results[question_id][0]  # Take the first answer

#         # Evaluate combined passage using AutoAIS
#         autoais_result = infer_autoais(question, combined_passage, answer)

#         # Display the results
#         print(f"\nQuestion (ID {question_id}): {question}\n")
#         print("Combined Passage:")
#         print(combined_passage)
#         print(f"\nAnswer: {answer}")
#         print(f"AutoAIS Evaluation: {autoais_result}\n")
    
#     evaluate_contexts(top_contexts_1, "Setting 1 (Q + A)")
#     evaluate_contexts(top_contexts_2, "Setting 2 (Q + A + C)")
#     evaluate_contexts(top_contexts_3_combined, "Setting 3 (3-step Q + A + C)")

# # Main function
# def main():
#     parser = argparse.ArgumentParser(description="Compare top retrieved contexts across different settings with AutoAIS evaluation.")
#     parser.add_argument("--setting1", required=True, help="Path to JSON file for Setting 1 (Q + A).")
#     parser.add_argument("--setting2", required=True, help="Path to JSON file for Setting 2 (Q + A + C).")
#     parser.add_argument(
#         "--setting3", 
#         required=True, 
#         nargs=3, 
#         help="Paths to 3 JSON files for Setting 3 (3-step Q + A + C)."
#     )
#     parser.add_argument("--inference_results", required=True, help="Path to inference results file (.pkl).")
#     parser.add_argument("--question_id", type=int, required=True, help="Question ID to analyze.")
#     parser.add_argument("--top_n", type=int, default=5, help="Number of top contexts to compare (default: 5).")

#     args = parser.parse_args()

#     inference_results = load_inference_results(args.inference_results)

#     compare_retrievals(
#         setting1=args.setting1,
#         setting2=args.setting2,
#         setting3_files=args.setting3,
#         inference_results=inference_results,
#         question_id=args.question_id,
#         top_n=args.top_n
#     )

# if __name__ == "__main__":
#     main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import argparse
import json
import pickle
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load AutoAIS model and tokenizer
AUTOAIS = "google/t5_xxl_true_nli_mixture"
hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS, device_map="auto")

def format_example_for_autoais(question, passage, answer):
    """Format example for AutoAIS evaluation."""
    input_text = f"premise: {passage} hypothesis: The answer to the question '{question}' is '{answer}'"
    return input_text

def infer_autoais(question, passage, answer):
    """Run AutoAIS inference."""
    input_text = format_example_for_autoais(question, passage, answer)
    input_ids = hf_tokenizer(input_text, return_tensors="pt").to("cuda").input_ids
    outputs = hf_model.generate(input_ids, max_length=1536)
    result = hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "Y" if result == "1" else "N"

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load inference results from .pkl file
def load_inference_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Extract question and top N retrieved passages
def extract_question_and_contexts(data, question_id, top_n=5):
    for item in data:
        if item['id'] == question_id:
            question = item['question']
            contexts = item['ctxs'][:top_n]
            return question, contexts
    return None, []

# Evaluate settings 1 and 2
def evaluate_single_setting(setting_file, inference_results, question_id, top_n):
    data = load_json(setting_file)
    question, contexts = extract_question_and_contexts(data, question_id, top_n)
    answer = inference_results[question]['answer']  # Take the first answer
    print(f"\n--- Evaluation for {setting_file} ---")
    print(f"Question (ID {question_id}): {question}\n")
    print(f"Answer: {answer}\n")
    for idx, ctx in enumerate(contexts):
        passage = ctx['text']
        autoais_result = infer_autoais(question, passage, answer)
        print(f"Passage {idx + 1}: {passage}")
        print(f"AutoAIS Evaluation: {autoais_result}\n")

# Evaluate combined diverse contexts for Setting 3
def evaluate_diverse_contexts(setting3_files, inference_results, question_id):
    diverse_passages = []
    question = None
    for file in setting3_files:

        data = load_json(file)
        question, contexts = extract_question_and_contexts(data, question_id, top_n=1)
        if contexts:
            diverse_passages.append(contexts[0]['text'])

    combined_passage = " ".join(diverse_passages)
    # question = inference_results[question][0]['question']
    answer = inference_results[question]['answer']  # Take the first answer

    autoais_result = infer_autoais(question, combined_passage, answer)

    print("\n--- Evaluation for Setting 3 (Diverse Contexts) ---")
    print(f"Question (ID {question_id}): {question}\n")
    print("Combined Passage:")
    print(combined_passage)
    print(f"\nAnswer: {answer}")
    print(f"AutoAIS Evaluation: {autoais_result}\n")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate Attributed QA Retrieval Settings.")
    parser.add_argument("--setting1", required=True, help="Path to JSON file for Setting 1 (Q + A).")
    parser.add_argument("--setting2", required=True, help="Path to JSON file for Setting 2 (Q + A + C).")
    parser.add_argument(
        "--setting3", 
        required=True, 
        nargs=3, 
        help="Paths to 3 JSON files for Setting 3 (diverse contexts)."
    )
    parser.add_argument("--inference_results", required=True, help="Path to inference results file (.pkl).")
    parser.add_argument("--question_id", type=int, required=True, help="Question ID to analyze.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top passages to evaluate per setting.")

    args = parser.parse_args()

    inference_results = load_inference_results(args.inference_results)

    # Evaluate Setting 1
    evaluate_single_setting(args.setting1, inference_results, args.question_id, args.top_n)

    # Evaluate Setting 2
    evaluate_single_setting(args.setting2, inference_results, args.question_id, args.top_n)

    # Evaluate Setting 3
    evaluate_diverse_contexts(args.setting3, inference_results, args.question_id)

if __name__ == "__main__":
    main()
