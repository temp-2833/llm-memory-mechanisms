import json
import re
from typing import List, Dict
from collections import defaultdict
import statistics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []

    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))

    common_tokens = pred_tokens & ref_tokens

    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def calculate_bleu(prediction: str, reference: str) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], pred_tokens,
                          weights=(1, 0, 0, 0),
                          smoothing_function=smoothing)
    return bleu1


def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main(file_path: str):
    data = load_data(file_path)

    category_metrics = defaultdict(lambda: {'f1': [], 'bleu': []})

    for sample in data:
        category = sample['category']
        system_answer = sample['system_answer']
        original_answer = sample['original_answer']

        f1 = calculate_f1(system_answer, original_answer)
        bleu1 = calculate_bleu(system_answer, original_answer)

        category_metrics[category]['f1'].append(f1)
        category_metrics[category]['bleu'].append(bleu1)

    for category, scores in sorted(category_metrics.items()):
        avg_f1 = statistics.mean(scores['f1'])
        avg_bleu = statistics.mean(scores['bleu'])
        count = len(scores['f1'])
        print(f"Category {category} (样本数: {count}):")
        print(f"  平均 F1 分数: {avg_f1:.4f}")
        print(f"  平均 BLEU-1 分数: {avg_bleu:.4f}")
        print()


if __name__ == "__main__":
    file_path = "A-Mem-locomo-results-all.json"
    main(file_path)