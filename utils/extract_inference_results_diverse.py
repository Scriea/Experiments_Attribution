"""
This script is used for extracting the inference results for the answer and context from the 
inference files for the Natural Questions, HOTPOT QA and Trivia QA datasets.

It also extracts the grounding input for the retrieving contexts from wikipedia corpus.
"""

import os
import pickle
import argparse
import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("-a", "--inference_file_ans", type=str, help="Path to the input file", required=True)
parser.add_argument("-c","--inference_file_context", type=str, help="Path to the input file", required=True)
parser.add_argument("-d", "--dataset", choices=["nq_open", "hotpotqa", "triviaqa"], required=True, help="Dataset Used")
parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory", default=".")

args = parser.parse_args()

SAVE_DIR = args.output_dir

if args.dataset == "nq_open":
    print("Extracting Inference Results for Natural Questions Dataset")
    with open(args.inference_file_ans, 'rb') as myfile:
        inference_results_ans = pickle.load(myfile)
    with open(args.inference_file_context, 'rb') as myfile:
        inference_results_ans_context = pickle.load(myfile)

    dict_inference_results={}
    
    for key, values in tqdm.tqdm(inference_results_ans_context.items(), total=len(inference_results_ans_context), desc="Extracting Inference Results"):
        dict_inference_results[key] = []
        for value in values.split("\n\n\n"):
            temp={}
            temp['answer'] = ""
            temp['context'] = ""
            if (len(value.split('###Answer:')) > 1 and len(value.split('###Context:')) > 1):
                temp['answer'] = value.split('###Answer:')[-1].split('###Context:')[0].strip()
            if (len(value.split('###Context:')) > 1):
                temp['context'] = value.split("###Context:")[-1].split("###END")[0].strip()
            dict_inference_results[key].append(temp)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_diverse_context_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    dict_grounding_inputs=[{}, {}, {}]
    for key, values in tqdm.tqdm(inference_results_ans_context.items(), desc="Extracting Grounding Input"):
        # dict_grounding_input[key]=[]
        for j, value in enumerate(values.split("\n\n\n")):
            if (j<3 and len(inference_results_ans[key].split('###Answer:')) > 1 or len(value.split('###Context:')) > 1):
                signal = key + "? " + inference_results_ans[key].split('###Answer:')[-1].split('###END')[0].strip() + ". " + value.split("###Context:")[-1].split("###END")[0].strip()
                dict_grounding_inputs[j][key] = signal

    for i, dict_grounding_input in enumerate(dict_grounding_inputs):
        with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_diverse_context_{i+1}_input.pkl'), 'wb') as f:
            pickle.dump(dict_grounding_input, f)

elif args.dataset == "hotpotqa":
    print("Extracting Inference Results for HOTPOT QA")
    with open(args.inference_file_ans, 'rb') as myfile:
        inference_results_ans = pickle.load(myfile)
    with open(args.inference_file_context, 'rb') as myfile:
        inference_results_ans_context = pickle.load(myfile)

    dict_inference_results={}
    for key, value in tqdm.tqdm(inference_results_ans.items(), total=len(inference_results_ans), desc="Extracting Inference Results"):
        temp={}
        temp['answer'] = ""
        temp['context'] = ""
        if (len(value.split('###Answer:')) > 1 and len(value.split('###Context:')) > 1):
            temp['answer'] = value.split('###Answer:')[-1].split('###Context:')[0].strip()
        if (len(value.split('###Context:')) > 1):
            temp['context'] = inference_results_ans_context[key].split("###Context:")[-1].split("###END")[0].strip()
        dict_inference_results[key] = temp
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_context_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    dict_grounding_input={}
    for key, value in tqdm.tqdm(inference_results_ans.items(), desc="Extracting Grounding Input"):
        dict_grounding_input[key]=""
        
        if (len(value.split('###Answer:')) > 1 or len(inference_results_ans_context[key].split('###Context:')) > 1):
            signal = key + "? " + value.split('###Answer:')[-1].split('###END')[0].strip() + ". " + inference_results_ans_context[key].split("###Context:")[-1].split("###END")[0].strip()
            dict_grounding_input[key] = signal

    with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_context_input.pkl'), 'wb') as f:
        pickle.dump(dict_grounding_input, f)

elif args.dataset == "triviaqa":
    print("Extracting Inference Results for Trivia QA")
    with open(args.inference_file_ans, 'rb') as myfile:
        inference_results_ans = pickle.load(myfile)
    with open(args.inference_file_context, 'rb') as myfile:
        inference_results_ans_context = pickle.load(myfile)

    dict_inference_results={}
    for key, value in tqdm.tqdm(inference_results_ans.items(), total=len(inference_results_ans), desc="Extracting Inference Results"):
        temp={}
        temp['answer'] = ""
        temp['context'] = ""
        if (len(value.split('###Answer:')) > 1 and len(value.split('###Context:')) > 1):
            temp['answer'] = value.split('###Answer:')[-1].split('###Context:')[0].strip()
        if (len(value.split('###Context:')) > 1):
            temp['context'] = inference_results_ans_context[key].split("###Context:")[-1].split("###END")[0].strip()
        dict_inference_results[key] = temp
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_context_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    dict_grounding_input={}
    for key, value in tqdm.tqdm(inference_results_ans.items(), desc="Extracting Grounding Input"):
        dict_grounding_input[key]=""
        
        if (len(value.split('###Answer:')) > 1 or len(inference_results_ans_context[key].split('###Context:')) > 1):
            signal = key + "? " + value.split('###Answer:')[-1].split('###END')[0].strip() + ". " + inference_results_ans_context[key].split("###Context:")[-1].split("###END")[0].strip()
            dict_grounding_input[key] = signal

    with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_context_input.pkl'), 'wb') as f:
        pickle.dump(dict_grounding_input, f)