# src/evaluation.py

import logging
import numpy as np
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
import torch

logger = logging.getLogger(__name__)

def precision_at_k(relevant_indices: List[int], retrieved_indices: List[int], k: int) -> float:
    """
    Computes Precision@k.
    """
    retrieved_at_k = retrieved_indices[:k]
    relevant_at_k = set(relevant_indices).intersection(set(retrieved_at_k))
    precision = len(relevant_at_k) / k
    return precision

def recall_at_k(relevant_indices: List[int], retrieved_indices: List[int], k: int) -> float:
    """
    Computes Recall@k.
    """
    retrieved_at_k = retrieved_indices[:k]
    relevant_at_k = set(relevant_indices).intersection(set(retrieved_at_k))
    recall = len(relevant_at_k) / len(relevant_indices) if relevant_indices else 0.0
    return recall

def mean_reciprocal_rank(relevant_indices_list: List[List[int]], retrieved_indices_list: List[List[int]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    for relevant_indices, retrieved_indices in zip(relevant_indices_list, retrieved_indices_list):
        rank = 0
        for idx, retrieved_idx in enumerate(retrieved_indices):
            if retrieved_idx in relevant_indices:
                rank = idx + 1
                break
        if rank > 0:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    mrr = np.mean(reciprocal_ranks)
    return mrr

def ndcg_at_k(relevant_indices: List[int], retrieved_indices: List[int], k: int) -> float:
    """
    Computes nDCG@k.
    """
    dcg = 0.0
    for i, idx in enumerate(retrieved_indices[:k]):
        if idx in relevant_indices:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_indices))))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return ndcg

def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Computes BLEU score between reference and hypothesis.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    return bleu_score

def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Computes ROUGE scores between reference and hypothesis.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def compute_bert_score(references: List[str], hypotheses: List[str]) -> Tuple[List[float], float]:
    """
    Computes BERTScore between references and hypotheses.
    """
    P, R, F1 = bert_score.score(hypotheses, references, lang='en', verbose=False)
    avg_f1 = torch.mean(F1).item()
    return F1.tolist(), avg_f1
