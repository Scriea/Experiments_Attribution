import pickle
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gfile", help="Path to grounded json file", required=True, type=str)
parser.add_argument("--rfile", help="Path to response pickle file", required=True, type=str)
# parser.add_argument("--csv", help="If you want csv output",default=False)
parser.add_argument("--output-file", help="Output pickle file name that should end with .pkl", required=True, type=str)
args = parser.parse_args()
path = args.gfile
autoais_format = {}

with open(args.rfile, 'rb') as f:
    response = pickle.load(f)
f.close()

with open(path) as f:
    grouding_file = json.load(f)
    for item in grouding_file:
        autoais_format[item['question']] = {
            'question': item['question'],
            'answer': response[item['question']]['answer'],
            'passage' : item['ctxs'][0]['text'],
            'attribution': ''
        }
f.close()

output_file = args.output_file
if not output_file.endswith(".pkl"):
    raise TypeError("Error! Output file should be a pickle file (.pkl)") 
# output_file = "autoais_format_" + ('BM25' if 'BM25' in os.path.basename(path) else 'Random')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'wb') as g:
    pickle.dump(autoais_format, g)

