import pickle
import argparse
import os
import re


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to file")

args = parser.parse_args()
path = os.path.abspath(args.file)
predictions = pickle.load(open(path, 'rb'))

groudingformat = {}
for key, value in predictions.items():
    groudingformat[key] = value['context']

output_file = "./grounding/Grounding_format_" + os.path.basename(path)
with open(output_file, 'wb') as f:
    pickle.dump(groudingformat, f)

