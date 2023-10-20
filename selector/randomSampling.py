import pickle
import random

## Dataset
icl_dataset = pickle.load(open('data/finetune_data_600_plus_url.pkl', 'rb'))

def get_demonstrations_random(icl_dataset, k:int=3)-> dict:
    prompt = ""
    demonstrations = random.sample( [i for i in range(len(icl_dataset))],k )
    for i, demo in enumerate(demonstrations):
        question = demo["question"]
        answer = demo["anwer"]
        context = demo["context"]
        prompt+= f"Question {i+1}: {question} \nAnswer: {answer}\n"
    return prompt