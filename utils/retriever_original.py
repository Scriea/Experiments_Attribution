#!/usr/bin/python3
import faiss
from datasets import load_dataset
import json
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
import pickle


PKL_NAME="grounding_files/Mpt-7B/dict_grounding_posthoc_nq_open_op_answer_input"
# cache_dir="/data-mount-2t/kowndinya-cache/"
cache_dir="kowndinya-cache"
model=SentenceTransformer("gtr-t5-xxl")
wiki_attribution_corpus=load_dataset("kowndinya23/wikipedia-attribution-corpus", cache_dir=cache_dir)
nq_open=load_dataset("nq_open", cache_dir=cache_dir)
idx=faiss.read_index("wikipedia-attribution-corpus-pretrained-gtr-flatip.index")
# co=faiss.GpuMultipleClonerOptions()
# co.shard=True
# search_index=faiss.index_cpu_gpu_list(idx, co=co, gpus=[0,1,2,3,4,5])
search_index=idx

with open(f'{PKL_NAME}.pkl', 'rb') as f:
    predictions = pickle.load(f)
queries=[]
for key, value in predictions.items():
    queries.append(value)
nq_val_emb=model.encode(queries, show_progress_bar=True, batch_size=8, convert_to_numpy=True)

D_val, I_val=search_index.search(nq_val_emb, 100)

validation_data=[]
N_validation=nq_open["validation"].num_rows
progress_bar=tqdm(range(N_validation))
for i in range(N_validation):
    validation_data.append({})
    validation_data[-1]["id"]=i
    validation_data[-1]["question"]=nq_open["validation"][i]["question"]
    validation_data[-1]["answers"]=nq_open["validation"][i]["answer"]
    validation_data[-1]["ctxs"]=[]
    for idx in I_val[i].tolist():
        url=wiki_attribution_corpus["train"][(idx)]["id"]
        title=url.split("/")[-1].split("#")[0]
        text=wiki_attribution_corpus["train"][idx]["text"]
        validation_data[-1]["ctxs"].append({
            "title": title,
            "text": text,
        })
    progress_bar.update(1)

# save validation_data as json
with open(f"datasets/{PKL_NAME}.json", "w") as f:
    json.dump(validation_data, f, indent=4)