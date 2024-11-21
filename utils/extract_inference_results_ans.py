import os
import pickle
import argparse
import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--inference_file", type=str, help="Path to the input file", required=True)
parser.add_argument("-d", "--dataset", choices=["nq_open", "hotpotqa", "triviaqa"], required=True, help="Dataset Used")
parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory", default=".")

args = parser.parse_args()

SAVE_DIR = args.output_dir

if args.dataset == "nq_open":
    print("Extracting Inference Results for Natural Questions")
    with open(args.inference_file, 'rb') as myfile:
        inference_results = pickle.load(myfile)

    dict_inference_results={}
    for key, value in tqdm.tqdm(inference_results.items(), total=len(inference_results), desc="Extracting Inference Results"):
        # print(key)
        temp={}
        temp['answer'] = ""
        temp['context'] = ""
        if (len(value.split('###Answer:')) > 1):
            temp['answer'] = value.split('###Answer:')[-1].split('###END')[0].strip()
        dict_inference_results[key] = temp
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    dict_grounding_input={}
    for key, value in tqdm.tqdm(inference_results.items(), desc="Extracting Grounding Input"):
        dict_grounding_input[key]=""
        # print(value)
        if (len(value.split('###Answer:')) > 1):
            dict_grounding_input[key] = key + "? " + value.split('###Answer:')[-1].split('###END')[0].strip()

    with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_input.pkl'), 'wb') as f:
        pickle.dump(dict_grounding_input, f)

elif args.dataset == "hotpotqa":
    with open(args.inference_file, 'rb') as myfile:
        inference_results = pickle.load(myfile)

    dict_inference_results={}
    for key, value in inference_results.items():
        # print(key)
        temp={}
        temp['answer'] = ""
        temp['context'] = ""
        if (len(value.split('###Answer:')) > 1):
            temp['answer'] = value.split('###Answer:')[-1].split('###END')[0].strip()
        dict_inference_results[key] = temp
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    dict_grounding_input={}
    for key, value in tqdm.tqdm(inference_results.items(), desc="Extracting Grounding Input"):
        dict_grounding_input[key]=""
        # print(value)
        if (len(value.split('###Answer:')) > 1):
            dict_grounding_input[key] = key + "? " + value.split('###Answer:')[-1].split('###END')[0].strip()

    with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_input.pkl'), 'wb') as f:
        pickle.dump(dict_grounding_input, f)

elif args.dataset == "triviaqa":
    with open(args.inference_file, 'rb') as myfile:
        inference_results = pickle.load(myfile)
    
    dict_inference_results={}
    for key, value in inference_results.items():
        # print(key)
        temp={}
        temp['answer'] = ""
        temp['context'] = ""
        if (len(value.split('###Answer:')) > 1):
            temp['answer'] = value.split('###Answer:')[-1].split('###END')[0].strip()
        dict_inference_results[key] = temp
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'dict_posthoc_{args.dataset}_op_answer_inference_results.pkl'), 'wb') as f:
        pickle.dump(dict_inference_results, f)

    
    dict_grounding_input={}
    for key, value in tqdm.tqdm(inference_results.items(), desc="Extracting Grounding Input"):
        dict_grounding_input[key]=""
        # print(value)
        if (len(value.split('###Answer:')) > 1):
            dict_grounding_input[key] = key + "? " + value.split('###Answer:')[-1].split('###END')[0].strip()

    with open(os.path.join(SAVE_DIR, f'dict_grounding_posthoc_{args.dataset}_op_answer_input.pkl'), 'wb') as f:
        pickle.dump(dict_grounding_input, f)