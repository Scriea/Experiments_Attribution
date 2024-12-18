import pickle
import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gfile", help="Path to grounded json file", required=True, type=str)
parser.add_argument("--rfile", help="Path to response pickle file", required=True, type=str)
parser.add_argument("--k", help="Top k passages to consider", required=True, type=int)
# parser.add_argument("--csv", help="If you want csv output",default=False)
parser.add_argument("--output-dir", help="Output directory name (this will contain k files)", required=True, type=str)
args = parser.parse_args()
path = args.gfile
# autoais_format = {}

with open(args.rfile, 'rb') as f:
    response = pickle.load(f)
f.close()

autoai5665  s_formats = [{} for i in range(args.k)]
# [{}]*k
with open(path) as f:
    grouding_file = json.load(f)
    
    for i in range(args.k):
        for item in grouding_file:
            autoais_formats[i][item['question']] = {
                'question': item['question'],
                'answer': response[item['question']]['answer'],
                'passage' : item['ctxs'][i]['text'],
                'attribution': ''
            }
f.close()

os.makedirs(args.output_dir, exist_ok=True)
# if not output_file.endswith(".pkl"):
#     raise TypeError("Error! Output file should be a pickle file (.pkl)") 
# output_file = "autoais_format_" + ('BM25' if 'BM25' in os.path.basename(path) else 'Random')
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
basename = os.path.basename(os.path.abspath(path).split('.')[0])
print(basename)
for i in range(args.k):
    outpath = os.path.join(args.output_dir, f"{basename}_top_{args.k}_index_{i + 1}.pkl")
    with open(outpath, 'wb') as g:
        pickle.dump(autoais_formats[i], g)
    g.close()
