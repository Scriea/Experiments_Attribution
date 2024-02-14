import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pickle
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from selector import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} {torch.cuda.current_device()}")
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
    # icl_dataset = pickle.load(open('../data/finetune_data_600_plus_url.pkl', 'rb'))
    nq_open = load_dataset('nq_open', cache_dir='../../data/')
    dev_data = nq_open['validation']
    
    instruction_prompt1 = "Given a question generate background context. \nExamples:\n"
    instruction_prompt2 = "Your task is to answer the given question based on the given context."

    demo_prompt1 = get_demonstrations_random_wiki(include_answers=False)
    demo_prompt2 = get_demonstrations_random_wiki(include_answers=True, have_context=True)
    final_results_random={}

    ## Two stage attribution
    for i in range(len(dev_data)):
        print(f"Executed {i}")
        query1 = f"{instruction_prompt1}\n{demo_prompt1}\nQuestion: {dev_data[i]['question']} ?\nContext:"
        print(dev_data[i]['question'])
        print("\n")
        sequences1 = pipeline(
            query1,
            max_length=1024,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_context  = sequences1[0]['generated_text'][len(query1)-8:]
        print(generated_context, end="\n\n")

        query2 = f"{instruction_prompt2} {demo_prompt2}\n{generated_context}\nQuestion: {dev_data[i]['question']}\nAnswer:"
        sequences2 = pipeline(
            query2,
            max_length=1024,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ans  = sequences2[0]['generated_text'][len(query2)-7:]
        print(f"{generated_ans}", end="\n\t---------------\t\n")


        res_ans = "".join(re.split("Answer:", generated_ans, flags=re.IGNORECASE))
        res_context = "".join(re.split("Context:", generated_context, flags= re.IGNORECASE))
        with open("test_nqopen_dev_random_wiki_2.txt", "a", encoding='utf-8') as myfile:
            myfile.write(sequences2[0]['generated_text'] + "\n\n\n")

        final_results_random[dev_data[i]['question']] = {'answer': res_ans, 'context' : res_context
                                    }
        with open('ICL_Random_Wiki_2_Results.pkl', 'wb') as f:
            pickle.dump(final_results_random, f)

if __name__ == "__main__":
    main()
