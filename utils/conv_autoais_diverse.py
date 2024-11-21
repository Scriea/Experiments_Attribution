import pickle
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gfile1", help="Path to grounded json files", required=True, type=str)
parser.add_argument("--gfile2", help="Path to grounded json files", required=True, type=str)
parser.add_argument("--gfile3", help="Path to grounded json files", required=True, type=str)    
parser.add_argument("--rfile", help="Path to response pickle file", required=True, type=str)
# parser.add_argument("--csv", help="If you want csv output",default=False)
parser.add_argument("--output-file", help="Output pickle file name that should end with .pkl", required=True, type=str)
args = parser.parse_args()
autoais_format = {}

with open(args.rfile, 'rb') as f:
    response = pickle.load(f)
f.close()

with open(args.gfile1) as f:
    grouding_file1 = json.load(f)
    
with open(args.gfile2) as f:
    grouding_file2= json.load(f)

with open(args.gfile3) as f:
    grouding_file3 = json.load(f)
    
for item in grouding_file1:
    autoais_format[item['question']] = {
        'question': item['question'],
        'answer': response[item['question']]['answer'],
        'passage' : item['ctxs'][0]['text'],
        'attribution': ''
    }
for item in grouding_file2:
    autoais_format[item['question']]['passage'] += "\n\n" + item['ctxs'][0]['text']
    
for item in grouding_file3:
    autoais_format[item['question']]['passage'] += "\n\n" + item['ctxs'][0]['text']


output_file = args.output_file

# output_file = "autoais_format_" + ('BM25' if 'BM25' in os.path.basename(path) else 'Random')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as g:
    pickle.dump(autoais_format, g)

