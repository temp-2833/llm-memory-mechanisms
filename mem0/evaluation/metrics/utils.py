"""
Borrowed from https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py

@article{xu2025mem,
    title={A-mem: Agentic memory for llm agents},
    author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao
           and Zhang, Yongfeng},
    journal={arXiv preprint arXiv:2502.12110},
    year={2025}
}
"""

import statistics
from collections import defaultdict
from typing import Dict, List, Union

import nltk
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from sentence_transformers.util import pytorch_cos_sim

# try:
#     nltk.download("punkt", quiet=True)
#     nltk.download("wordnet", quiet=True)
# except Exception as e:
#     print(f"Error downloading NLTK data: {e}")

# 初始化嵌入模型
try:
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"警告: 无法加载SentenceTransformer模型: {e}")
    sentence_model = None


def simple_tokenize(text):
    # 简单的分词器
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    # 根据prediction和正确答案计算rouge分数
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    # 计算BLEU分数
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            score = 0.0
        scores[f"bleu{n}"] = score

    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    # 为了余弦相似度计算BERTScore
    try:
        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    # 计算METEOR
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    # 计算句子嵌入相似度
    if sentence_model is None:
        return 0.0
    try:
        # 编码
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)

        # 计算余弦相似度
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    # 计算所有指标
    # 处理空值
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

    # 转换成string
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # 计算精准匹配
    exact_match = int(prediction.lower() == reference.lower())

    # 计算基于token的F1分数
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 计算BLEU分数
    bleu_scores = calculate_bleu_scores(prediction, reference)

    # 组合所有指标
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }

    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]], all_categories: List[int]
) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    # 以category分类，计算整体指标
    if not all_metrics:
        return {}

    # 初始化
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    # 收集所有指标值
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    # 总体指标
    results = {"overall": {}}

    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    # 为每个category计算统计量
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return results
