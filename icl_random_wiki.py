import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pickle
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from selector import *

"""
Helper Functions
"""
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
   # icl_dataset = pickle.load(open('data/finetune_data_600_plus_url.pkl', 'rb'))
    nq_open = load_dataset('nq_open', cache_dir='../data/')
    dev_data = nq_open['validation']
    initial_string = "Given a question generate background context and answer the given question based on the generated context. Ignore <EOE>.\nExamples:\n"
    demo_prompt = get_demonstrations_random_wiki()
    final_results_random={}

    ## RandomSampling
    print("Method: Random Sampling")
    for i in range(len(dev_data)):
        print("Executed " + str(i))
        query = initial_string + demo_prompt + "\nQuestion:" + dev_data[i]['question']
        sequences = pipeline(
            query,
            max_length=1024,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            res = seq['generated_text'][len(query):]
            res = re.split("Answer:", res, flags=re.IGNORECASE)
            res_context = "".join(re.split("Output:", res[0], flags=re.IGNORECASE)[1:]).replace("{","").replace("{","").replace("Context:","")
            res_ans = res[1].split("Question:")[0].replace("{","").replace("}","") if len(res)>1 else ""
            print(f"Result:\n{seq['generated_text']}")
            print(10*"--","\n")
            print(res_context)
            print(res_ans)
            print(10*"--","\n")	

            with open("test_nqopen_dev_random_wiki.txt", "a", encoding='utf-8') as myfile:
                myfile.write(seq['generated_text'] + "\n")
            final_results_random[dev_data[i]['question']] = {'answer': res_ans, 'context' : res_context
                                        }
            with open('ICL_Random_Wiki_Results.pkl', 'wb') as f:
                pickle.dump(final_results_random, f)

if __name__ == "__main__":
    main()
