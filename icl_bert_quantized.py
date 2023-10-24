import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import pickle
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from bert_score import score
import numpy as np

def main():

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        #bnb_4bit_quant_type="nf4",
        #bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map = "auto",
    )
    icl_dataset = pickle.load(open('data/finetune_data_600_plus_url.pkl', 'rb'))
    nq_open = load_dataset('nq_open', cache_dir='../data/')
    # dev_data = nq_open['validation']
    train_data = nq_open['train']
    initial_string = "Given a question generate background context and answer the given question based on the generated context.\nExamples:"
    final_results_bert={}
    documents = [item['question'] for item in icl_dataset]
 
    ## RandomSampling
    print("Method: BERTScore")
    for i in range(len(train_data)):
        print("Executed " + str(i))
        demo_prompt = get_demonstrations_bert(icl_dataset, 3)
        query = initial_string + demo_prompt + "\nQuestion:" + train_data[i]['question'] + "\nOutput: "
        sequences = pipeline(
            query,
            max_length=512,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            res = " ".join(re.split("Output:",seq['generated_text'].split("<EOE>")[1], flags=re.IGNORECASE))
            res = re.split("Answer:", res, flags=re.IGNORECASE)
            res_context = "".join(res[0].split(train_data[i]['question'])[1:]).strip()
            res_ans = res[1].split("Question:")[0].strip() if len(res)>1 else res_context
            print(f"Result:\n{seq['generated_text']}")
            with open("test_nqopen_bert.txt", "w", encoding='utf-8') as myfile:
                myfile.write(seq['generated_text'] + "\n")
            final_results_bert[train_data[i]['question']] = {
                                        'answer': res_ans,
                                        'context' : res_context
                                        }
            with open('ICL_bert_results.pkl', 'wb') as f:
                pickle.dump(final_results_bert, f)

if __name__ == "__main__":
    main()

