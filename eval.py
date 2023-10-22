import collections
import os
import re
import string
import numpy as np

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def compute_EM(predictions:list, references:list):
        ##if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)
        # if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)
        score_list = predictions == references
        return np.mean(score_list)

def computer_f1(predictions:list, references:list):
        pass

def squad_score(predictions:list, references:list):
        score = {}
        score['exact_match': compute_EM(predictions, references)]