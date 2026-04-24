import argparse
import concurrent.futures
import json
import threading
from collections import defaultdict
# -------------------------
import nltk, os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'

nltk.download = lambda *args, **kwargs: None

nltk_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path = [nltk_dir]
print(f"NLTK数据路径: {nltk_dir}")

try:
    nltk.data.find('tokenizers/punkt')
    print("✓ NLTK使用本地数据成功")
    from nltk.tokenize import word_tokenize
    test_text = "Hello world"
    tokens = word_tokenize(test_text)
    print(f"✓ NLTK分词测试成功: {tokens}")
except LookupError as e:
    print(f"❌ NLTK数据缺失: {e}")

from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm


def process_item(item_data):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        # Skip category 5
        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        # llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                # "llm_score": llm_score,
            }
        )

    return local_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    # 输入文件
    parser.add_argument(
        "--input_file", type=str, default="results/mem0_results_top_6_filter_False_graph_False.json", help="Path to the input dataset file"
    )
    # 输出文件
    parser.add_argument(
        "--output_file", type=str, default="evaluation_metrics.json", help="Path to save the evaluation results"
    )

    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker threads")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()  # 线程锁
    # 使用线程池并行处理数据
    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in data.items()]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            local_results = future.result()
            with results_lock:
                for k, items in local_results.items():
                    results[k].extend(items)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"结果已保存到 {args.output_file}")


if __name__ == "__main__":
    main()
