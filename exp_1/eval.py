import collections
import os
import re
import string
import numpy as np
import pickle
from datasets import load_dataset

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
        def remove_articles(text):
                return ARTICLES_REGEX.sub(" ", text)
        def white_space_fix(text):
                return " ".join(text.split())
        def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
                return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
        if not s:
                return []
        return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
                # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
                return int(gold_toks == pred_toks)
        if num_same == 0:
                return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def get_scores(references, predictions):
        if not len(predictions)== len(references):
                print("Dimensions not same")
        else:
                em_scores=[]
                f1_scores=[]
                for i in range(len(predictions)):
                        a_pred = predictions[i]
                        a_gold = references[i]
                        em_scores.append(compute_exact(a_gold=a_gold, a_pred=a_pred))
                        f1_scores.append(compute_f1(a_gold=a_gold, a_pred=a_pred))

                return {'exact_match': np.mean(em_scores)*100, 'f1': np.mean(f1_scores)*100}

if __name__ == "__main__":
        icl_bm25_results = pickle.load(open('ICL_BM25_results.pkl', 'rb'))
        icl_random_results = pickle.load(open('ICL_Random_Sampling_dev_results.pkl', 'rb'))
        nq_open = load_dataset('nq_open')
        dev_data = nq_open['validation']
        gold_answers = ["".join(answer) for answer in dev_data['answer']]
        bm25_predictions = [icl_bm25_results[question]['answer'] for question in dev_data['question']]
        random_predictions = [icl_random_results[question]['answer'] for question in dev_data['question']]


        bm25_score = get_scores(gold_answers, bm25_predictions)
        random_score = get_scores(gold_answers, random_predictions)

        print("Method: Random Sampling")
        print(random_score)

        print("BM25 Sampling")
        print(bm25_score)

"""
def get_raw_scores(references:list, preds:list):
        exact_scores = {}
        f1_scores = {}
        for article in dataset:
                for p in article["paragraphs"]:
                for qa in p["qas"]:
                        qid = qa["id"]
                        gold_answers = [t for t in qa["answers"]["text"] if normalize_answer(t)]
                        if not gold_answers:
                        # For unanswerable questions, only correct answer is empty string
                        gold_answers = [""]
                        if qid not in preds:
                        print(f"Missing prediction for {qid}")
                        continue
                        a_pred = preds[qid]
                        # Take max over all gold answers
                        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        return exact_scores, f1_scores

def compute_EM(predictions:list, references:list):
        predictions = np.asarray(predictions)
        references = np.asarray(references)
        ## ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)
        # ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)
        score_list = predictions == references
        return np.mean(score_list)

"""
