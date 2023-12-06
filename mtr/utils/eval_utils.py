import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from astropy.stats import jackknife

def _text_representation(args, text, tokenizer):
    if args.text_type == "bert":
        if isinstance(text, str):
            text_inputs = tokenizer(text, return_tensors="pt")['input_ids']
        elif isinstance(text, list):
            text_inputs = tokenizer(", ".join(text), return_tensors="pt")['input_ids']
    elif args.text_type == "glove":
        if isinstance(text, str):
            text_inputs = torch.from_numpy(np.array(tokenizer[text]).astype("float32")).unsqueeze(0)
        elif isinstance(text, list):
            tag_embs= np.array([tokenizer[tag] for tag in text]).astype("float32")
            text_inputs = torch.from_numpy(tag_embs.mean(axis=0)).unsqueeze(0)
    else:
        print("error!")
    return text_inputs 
    
def single_query_evaluation(targets, logits, labels):
    """
    target = Pandas DataFrame Binary Mtrix( track x label )
    logits = Pandas DataFrame Logit Mtrix( track x label )
    label = tag list
    """
    targets = targets[labels].values
    logits = logits[labels].values
    roc_auc = metrics.roc_auc_score(targets, logits, average='macro')
    pr_auc = metrics.average_precision_score(targets, logits, average='macro')
    results = {
        'roc_auc' :roc_auc,
        'pr_auc': pr_auc
    }
    print(f"roc_auc: {roc_auc}, pr_auc: {pr_auc}")
    # tag wise score
    roc_aucs = metrics.roc_auc_score(targets, logits, average=None)
    pr_aucs = metrics.average_precision_score(targets, logits, average=None)
    tag_wise = {}
    for i in range(len(labels)):
        tag_wise[labels[i]] = {
            "roc_auc":roc_aucs[i], 
            "pr_auc":pr_aucs[i]
    }
    results['tag_wise'] = tag_wise
    results['count'] = len(labels)
    return results
    
def retrieval_evaluation(audio_embs, text_embs, captions, songs, cluster_mask=None):
    results = dict()
    # Text to audio retrieval
    logits = text_embs @ audio_embs.T
    predictions = {}
    for i in range(logits.shape[0]):
        sorted_idx = torch.argsort(logits[i], descending=True)
        sorted_song = [songs[idx] for idx in sorted_idx]
        predictions[captions[i]] = sorted_song
    results['text_to_audio'] = rank_eval(captions, songs, predictions)
    print("Text to audio: ", results['text_to_audio'])

    # Audio to text retrieval
    logits = audio_embs @ text_embs.T
    predictions = {}
    for i in range(logits.shape[0]):
        sorted_idx = torch.argsort(logits[i], descending=True)
        sorted_caption = [captions[idx] for idx in sorted_idx]
        predictions[songs[i]] = sorted_caption
    results['audio_to_text'] = rank_eval(songs, captions, predictions)
    print("Audio to text: ", results['audio_to_text'])
    
    caption_similarity = text_embs @ text_embs.T
    results['caption_stat'] = {
        "mean": float(caption_similarity.mean()),
        "std": float(caption_similarity.std())
    }
    audio_similarity = audio_embs @ audio_embs.T
    results['audio_stat'] = {
        "mean": float(audio_similarity.mean()),
        "std": float(audio_similarity.std())
    }
    return results

def rank_eval(sources, targets, pred_items):
    """
        captions = Dict: caption -> msdid
        pred_items = Dict: caption -> [msdid, msdid, msdid, ...] sort by relevant score
    """
    R1, R5, R10, mAP10, med_rank = [], [], [], [], []
    for i, source in enumerate(sources):
        target = targets[i]
        pred_names = pred_items[source]
        preds = np.array([target == pred for pred in pred_names])
        rank_value = np.nonzero(preds)[0][0]
        
        R1.append(float(rank_value < 1))
        R5.append(float(rank_value < 5))
        R10.append(float(rank_value < 10))
        
        positions = [rank_value + 1] if rank_value < 10 else []
        if len(positions) > 0:
            precisions = np.divide(np.arange(1, len(positions) + 1, dtype=float), positions)
            avg_precision = np.sum(precisions, dtype=float)
            mAP10.append(avg_precision)
        else:
            mAP10.append(0.0)
        
        med_rank.append(rank_value)

    r1_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R1), np.mean, 0.95)
    r5_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R5), np.mean, 0.95)
    r10_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(R10), np.mean, 0.95)
    map_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(mAP10), np.mean, 0.95)
    medrank_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(med_rank), np.median, 0.95)
    return {
        "R@1": r1_estimate,
        "R@5": r5_estimate,
        "R@10": r10_estimate,
        "mAP@10": map_estimate,
        "medRank": medrank_estimate,
    }

def _medrank(logits):
    rank_values = []
    for idx in range(len(logits)):
        _, idx_list = logits[idx].topk(len(logits))
        rank_value = float(torch.where(idx_list == idx)[0])
        rank_values.append(float(rank_value))
    medrank_estimate, _, _, _ = jackknife.jackknife_stats(np.asarray(rank_values), np.median, 0.95)
    return medrank_estimate


def tag_to_binary(tag_list, list_of_label, tag_to_idx):
    bainry = np.zeros([len(list_of_label),], dtype=np.float32)
    for tag in tag_list:
        bainry[tag_to_idx[tag]] = 1.0
    return bainry

def _continuous_target(binary):
    sum_= binary.sum(dim=1)
    mask = sum_ - torch.matmul(binary, binary.T)
    inverse_mask = 1 / mask
    continuous_label = torch.nan_to_num(inverse_mask, posinf=.0)
    return continuous_label