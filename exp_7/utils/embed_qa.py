import argparse
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def main(json_file, pickle_file, output_questions_file, output_answers_file, output_question_plus_answer_file):
    # Load the JSON file
    with open(json_file, 'r') as f:
        retrieval_results = json.load(f)

    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        qa_dict = pickle.load(f)

    # Ensure the SentenceTransformer model is available
    model = SentenceTransformer("gtr-t5-xxl")
    model.half()
    print("Model Loaded")

    # Extract questions and answers in JSON order
    questions = []
    answers = []
    question_plus_answer = []
    for item in retrieval_results:
        question_text = item['question']
        if question_text in qa_dict:
            questions.append(question_text)
            answers.append(qa_dict[question_text]['answer'])
            question_plus_answer.append(question_text + "? " + qa_dict[question_text]['answer'])  
        else:
            print(f"Warning: Question '{question_text}' not found in pickle file.")
            exit()

    # Embed questions and answers
    print("Embedding questions...")
    # question_embeddings = model.encode(questions, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

    print("Embedding answers...")
    # answer_embeddings = model.encode(answers, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    
    print("Embedding question + answers...")
    question_plus_answer_embeddings = model.encode(question_plus_answer, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    # Save the embeddings to .npy files
    
    # np.save(output_questions_file, question_embeddings)
    # np.save(output_answers_file, answer_embeddings)
    np.save(output_question_plus_answer_file, question_plus_answer_embeddings)
    print(f"Question embeddings saved to {output_questions_file}")
    print(f"Answer embeddings saved to {output_answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save embeddings for questions and answers.")
    parser.add_argument("--json_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--inf_file", type=str, help="Path to the input pickle file.")
    parser.add_argument("--output_questions_file", type=str, help="Path to save the question embeddings (.npy).", default="./embedding_questions.npy")
    parser.add_argument("--output_answers_file", type=str, help="Path to save the answer embeddings (.npy).", default="./embedding_answers.npy")
    parser.add_argument("--output_question_plus_answer_file", type=str, help="Path to save the question + answer embeddings (.npy).", default="./embedding_question_plus_answer.npy")
    
    args = parser.parse_args()

    main(args.json_file, args.inf_file, args.output_questions_file, args.output_answers_file, args.output_question_plus_answer_file)
