# 离线评估脚本，对all_loco_results.json，计算每个category的平均F1分数
import json
import re
from typing import List, Dict
from collections import defaultdict
import statistics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def simple_tokenize(text: str) -> List[str]:
    # 简单版的分词
    if not text:
        return []

    # 转换成string格式，转小写
    text = str(text).lower()
    # 去掉标点、用空格分割 using regex
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_f1(prediction: str, reference: str) -> float:

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))

    # 计算交集
    common_tokens = pred_tokens & ref_tokens

    # 计算准确率和召回率
    precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0

    # 计算F1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def calculate_bleu(prediction: str, reference: str) -> float:
    # 计算BLEU-1分数，使用SmoothingFunction来处理预测或参考为空的情况
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)

    # 如果预测或参考为空，BLEU为0
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 使用平滑方法1，避免出现警告，并处理短句
    smoothing = SmoothingFunction().method1
    # weights=(1, 0, 0, 0) 表示只计算1-gram，即BLEU-1
    bleu1 = sentence_bleu([ref_tokens], pred_tokens,
                          weights=(1, 0, 0, 0),
                          smoothing_function=smoothing)
    return bleu1


def load_data(file_path: str) -> List[Dict]:
    # 从json文件加载数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main(file_path: str):
    # 按category分组，计算平均F1
    data = load_data(file_path)

    # 初始化category字典
    # category_f1 = defaultdict(list)
    category_metrics = defaultdict(lambda: {'f1': [], 'bleu': []})

    # 为每个样本计算分数
    for sample in data:
        category = sample['category']
        system_answer = sample['system_answer']
        original_answer = sample['original_answer']

        # 算
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
    file_path = "mem0_results_top_30-all.json"
    main(file_path)
