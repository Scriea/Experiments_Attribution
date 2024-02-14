import pickle
import os
import numpy
import json
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Path to grounded json file", required=True, type=str)
parser.add_argument("--csv", help="If you want csv output",default=False)
args = parser.parse_args()
path = args.file
autoais_format = {}
with open(path) as f:
    grouding_file = json.load(f)
    for item in grouding_file:
        autoais_format[item['question']] = {
            'question': item['question'],
            'answer': item['answers'][0],
            'passage' : item['ctxs'][0]['text'],
            'attribution': ''
        }
f.close()


output_file = "autoais_format_" + ('BM25' if 'BM25' in os.path.basename(path) else 'Random')

if args.csv:
    output_file+= '.csv'
    with open(output_file, 'w') as g:
        writer = csv.DictWriter(g, fieldnames=["question", "answer", "passage", "attribution"])
        writer.writeheader()
        for example in autoais_format.values():
            row = {
                "question": example["question"],
                "answer": example["answer"],
                "passage": example["passage"],
                "attribution": ''
            }
        writer.writerow(row)
else:
    output_file+= '.pkl'
    with open(output_file, 'wb') as g:
        
        pickle.dump(autoais_format, g)

