import pickle
from tqdm import tqdm
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import ConcaveOverModularFunction
from submodlib import LogDeterminantMutualInformationFunction
from submodlib_cpp import ConcaveOverModular
from submodlib import FacilityLocationMutualInformationFunction
from submodlib import GraphCutMutualInformationFunction


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSS Indexer")
    parser.add_argument(
        "--passage_embeddings", type=str, help="Path to passage embeddings"
    )
    parser.add_argument("--query_embeddings", type=str, help="Path to query embeddings")
    parser.add_argument(
        "--output", type=str, help="Path to output Directory", required=True
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.passage_embeddings, "rb") as f:
        retrieved_passage_embeddings = pickle.load(f)


    target_query_embeddings = np.load(args.query_embeddings)
    index_list = {}
    for i in tqdm(range(0, len(retrieved_passage_embeddings), 100), desc="Processing"):
        index = int(i / 100)
        targe_query = target_query_embeddings[index].reshape((1, 768))
        # query = qq[index].reshape((1, 768))
        qd = np.concatenate([targe_query])
        indices = []
        features = retrieved_passage_embeddings[i : i + 100]
        objFL = GraphCutMutualInformationFunction(
            n=100, num_queries=len(qd), data=features, queryData=qd, metric="cosine"
        )
        greedyList = objFL.maximize(
            budget=3,
            optimizer="NaiveGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
        )
        indices.append([i[0] for i in greedyList])
        index_list[index] = indices

  

    with open(os.path.join(args.output, "GraphCutMutualInformationFunction.json"), "w") as f:
        json.dump(index_list, f)

    index_list = {}
    for i in tqdm(range(0, len(retrieved_passage_embeddings), 100)):
        index = int(i / 100)
        targe_query = target_query_embeddings[index].reshape((1, 768))
        # query = qq[index].reshape((1, 768))
        qd = np.concatenate([targe_query])
        indices = []
        features = retrieved_passage_embeddings[i : i + 100]
        objFL = FacilityLocationMutualInformationFunction(
            n=100, 
            num_queries=len(qd), 
            data=features, 
            queryData=qd, 
            metric="cosine",
            magnificationEta=1.0
        )
        greedyList = objFL.maximize(
            budget=3,
            optimizer="NaiveGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
        )
        indices.append([i[0] for i in greedyList])
        index_list[index] = indices

    with open(os.path.join(args.output, "FacilityLocationMutualInformationFunction.json"), "w") as f:
        json.dump(index_list, f)


    index_list = {}
    for i in tqdm(range(0, len(retrieved_passage_embeddings), 100)):
        index = int(i / 100)
        targe_query = target_query_embeddings[index].reshape((1, 768))
        # query = qq[index].reshape((1, 768))
        qd = np.concatenate([targe_query])
        indices = []
        features = retrieved_passage_embeddings[i : i + 100]
        objFL = LogDeterminantMutualInformationFunction(
            n=100, 
            num_queries=len(qd), 
            data=features, 
            queryData=qd, 
            metric="cosine",
            magnificationEta=1.0,
            lambdaVal=0.5
        )
        greedyList = objFL.maximize(
            budget=3,
            optimizer="NaiveGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
        )
        indices.append([i[0] for i in greedyList])
        index_list[index] = indices

    with open(os.path.join(args.output, "LogDeterminantMutualInformationFunction.json"), "w") as f:
        json.dump(index_list, f)

    index_list = {}
    for i in tqdm(range(0, len(retrieved_passage_embeddings), 100)):
        index = int(i / 100)
        targe_query = target_query_embeddings[index].reshape((1, 768))
        # query = qq[index].reshape((1, 768))
        qd = np.concatenate([targe_query])
        indices = []
        features = retrieved_passage_embeddings[i : i + 100]
        objFL = ConcaveOverModularFunction(
            n=100, 
            num_queries=len(qd), 
            data=features, 
            queryData=qd, 
            metric="cosine",
        )
        greedyList = objFL.maximize(
            budget=3,
            optimizer="NaiveGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False,
        )
        indices.append([i[0] for i in greedyList])
        index_list[index] = indices

    with open(os.path.join(args.output, "ConcaveOverModularFunction.json"), "w") as f:
        json.dump(index_list, f)

