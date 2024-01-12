import pickle
import os
import re
import numpy
import json
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--grounded-file", help="Path to grounded json file", required=True, type=str)
parser.add_argument("--predictions", help="Path to your predictions", required= True, type= str)
parser.add_argument("--csv", help="If you want csv output",default=False)
args = parser.parse_args()
path = args.grounded_file
path_prediction = args.predictions
autoais_format = {}
predictions = pickle.load(open(path_prediction, 'rb'))
with open(path) as f:
    grouding_file = json.load(f)
    for item in grouding_file:
        question = item['question']
        predicted_answer = predictions[question]['answer']
        if 'context' in predicted_answer.lower():
            predicted_answer = ''
        autoais_format[question] = {
            'question': question,
            'answer': predicted_answer,
            'passage' : item['ctxs'][0]['text'],
            'attribution': ''
        }
f.close()


output_file = "autoais_format_Random_Wiki"

if args.csv:
    output_file+= '.csv'
    with open(output_file, 'w') as g:
        writer = csv.DictWriter(g, fieldnames=["question", "answer", "passage", "attribution"])
        writer.writeheader()
        for example in autoais_format.values():
            row = {
                "question": example["question"],
                "answer": example['answer'],
                "passage": example["passage"],
                "attribution": ''
            }
        writer.writerow(row)
else:
    output_file+= '.pkl'
    with open(output_file, 'wb') as g:
        pickle.dump(autoais_format, g)

