import random

def get_demonstrations_random(icl_dataset:list, k:int=3)-> dict:
    prompt = ""
    demonstrations = random.sample( [i for i in range(len(icl_dataset))],k )
    for i, demo in enumerate(demonstrations):
        question = demo["question"]
        answer = demo["anwer"]
        context = demo["context"]
        prompt+= f"Question {i+1}: {question} \nAnswer: {answer}\n"
    return prompt