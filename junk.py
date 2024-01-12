import pickle
import json
from datasets import load_dataset

nq_open = load_dataset('nq_open', cache_dir='../data/')
itm = pickle.load(open("results/ICL_Random_Wiki_Results.pkl", 'rb'))
dev_data = nq_open['validation']
   

with open("Wiki_Analysis.txt","w") as f:
    for i,key in enumerate(itm.keys()):
        # print(key , itm[key])
        li  = f"Question: {key}\nContext:{itm[key]['context']}\nAnswer_Output:{itm[key]['answer']}\nAnswer_Gold:{dev_data[i]['answer']}\n\n"
        f.writelines(li)