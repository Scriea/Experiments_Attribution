from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer
import transformers
import ast
import pickle


def main():

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        #bnb_4bit_quant_type="nf4",
        #bnb_4bit_compute_dtype=torch.float16,
    )

    #peft_model_id = "results_peft"
    #config = LoraConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    #model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

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

    #with open('our_alpaca_dev.json', 'r') as myfile:
    #    dev_data = myfile.read()

    nq_open = load_dataset('nq_open', cache_dir='../data/')
    dev_data = nq_open['validation']

    #dev_data = ast.literal_eval(dev_data)
    initial_string = "Generate a background context from Wikipedia to answer the following question. Question: "

    final_results={}
    for i in range(len(dev_data)):
        print("Executed " + str(i))
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

        


if __name__ == "__main__":
    main()
