import sys
from pathlib import Path
import json
from config import CLUSTERS


def tag_auc(k):
    with open("../exp/result/disentangle_32_best_cluster_results.json", "r") as f:
        results = json.load(f)
    
    for cluster in CLUSTERS:
        roc_results = []
        pr_results = []
        for tag in results[cluster]['tag_wise']:
            roc = results[cluster]['tag_wise'][tag]['roc_auc']
            roc_results.append((tag, roc))
            pr = results[cluster]['tag_wise'][tag]['pr_auc']
            pr_results.append((tag, pr))
        roc_results.sort(key=lambda x: x[1], reverse=True)
        pr_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"===================={cluster}====================")
        # print("roc_auc")
        # print(roc_results[:k])
        # print(roc_results[-k:])

        print("pr_auc")
        print(pr_results[:k])
        print(pr_results[-k:])

tag_auc(3)
