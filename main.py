import os
import pickle
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from selector import get_demonstrations_random

from evaluate import load
squad_metric = load("squad")


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
        device_map="auto",
    )
    icl_dataset = pickle.load(open('data/finetune_data_600_plus_url.pkl', 'rb'))
    nq_open = load_dataset('nq_open', cache_dir='../data/')
    dev_data = nq_open['validation']
    train_data = nq_open['train']
    initial_string = "Generate a background context to answer the following question. Here are few examples:"
    final_results={}

    for i in range(len(dev_data)):
        # print("Executed " + str(i))
        demonstrations = get_random_demonstrations(dev_data[i]['question'])
        query = initial_string + dev_data[i]['question'] + " Response: "
        sequences = pipeline(
            query,
            max_length=500,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
            with open("test_nqopen.txt", "a") as myfile:
                myfile.write(seq['generated_text'] + "\n")

        final_results[dev_data[i]['question']] = seq['generated_text']

        with open('llm_as_retriever_input_qnplusanswer_600_op_url_results.pkl', 'wb') as f:
                pickle.dump(final_results, f)

    results = squad_metric.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    main()

