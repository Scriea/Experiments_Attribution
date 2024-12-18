import pandas as pd

import numpy as np

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--csv", help="Autoais result csv file", required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    # print(df.info())
    df['context_length'] = df['passage'].apply(lambda x: len(x.split()))
    df_yes = df[df['autoais'] == 'Y']
    print(df_yes.index)
    df_no = df[df['autoais'] == 'N']
    print("Autoais count: ", len(df))
    print("Autoais Yes count: ", len(df_yes))
    print("Autoais No count: ", len(df_no))
    print("Autoais Score: ", len(df_yes)/len(df))
    

    ## Avg context length
    print("Avg context length: ", df['context_length'].mean())

    ## Avg context length for Yes and No
    print("Avg context length for Yes: ", df_yes['context_length'].mean())
    print("Avg context length for No: ", df_no['context_length'].mean())

    print("Avg context length for Yes: ", df_yes['context_length'].min())
    print("Avg context length for No: ", df_no['context_length'].min())
    print() 
    # print("###\n"*3)
    # for i in range(10):
    #     print("Question: ", df_no.iloc[i]['question'])
    #     print("Passage: ", df_no.iloc[i]['passage'])
    #     print("Answer: ", df_no.iloc[i]['answer'])
    #     print("Autoais: ", df_no.iloc[i]['autoais'])
    #     print("\n\n")





 