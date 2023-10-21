import random
from rank_bm25 import BM25Okapi

def get_demonstrations_random(icl_dataset:list, k:int=3)-> str:
    prompt = ""
    demonstrations = random.sample([i for i in range(len(icl_dataset))], k)
    for i in demonstrations:
        demo = icl_dataset[i]
        question = demo["question"]
        answer = "".join(demo["answer"])
        context = demo["context"]
        prompt+= f"Question: {question}\nContext: {context}\nAnswer: {answer}\n"
    return prompt

def get_demonstrations_coverage(icl_dataset:list, k:int=3)-> str:
    prompt = ""
    demonstrations = random.sample([i for i in range(len(icl_dataset))], k)
    for i in demonstrations:
        demo = icl_dataset[i]
        question = demo["question"]
        answer = demo["anwer"]
        context = demo["context"]
        prompt+= f"Question: {question}\nContext: {context}\nAnswer: {answer}\n"
    return prompt

def get_demonstrations_bm25(icl_dataset:list, corpus, bm25, query:str, k:int=3)-> str:
    prompt = ""
    tokenized_query = query.split(" ")
    demonstrations = bm25.get_top_n(tokenized_query, corpus, n=k)
    for i in demonstrations:
        demo = icl_dataset[i]
        question = demo["question"]
        answer = demo["anwer"]
        context = demo["context"]
        prompt+= f"Question {i+1}: {question}\nContext: {context}\nAnswer: {answer}\n"
    prompt += "```"
    return prompt
