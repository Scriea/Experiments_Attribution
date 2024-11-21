import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to file")

args = parser.parse_args()
path = os.path.abspath(args.file)
predictions = pickle.load(open(path, 'rb'))
print(predictions)


