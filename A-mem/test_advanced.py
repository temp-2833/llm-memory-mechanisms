import nltk
nltk.download = lambda *args, **kwargs: None  # 彻底禁用下载
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
API_KEY = ""
BASE_URL = ""

nltk_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path = [nltk_dir]
print(f"NLTK数据路径: {nltk_dir}")
# 验证NLTK是否工作
try:
    nltk.data.find('tokenizers/punkt')
    print("✓ NLTK使用本地数据成功")
    from nltk.tokenize import word_tokenize
    test_text = "Hello world"
    tokens = word_tokenize(test_text)
    print(f"✓ NLTK分词测试成功: {tokens}")
except LookupError as e:
    print(f"❌ NLTK数据缺失: {e}")

from memory_layer import LLMController, AgenticMemorySystem
from load_dataset import load_locomo_dataset, QA, Turn, Session, Conversation
from utils import calculate_metrics, aggregate_metrics

import json
import argparse
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
import statistics
from collections import defaultdict
import pickle
import random
from tqdm import tqdm
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize SentenceTransformer model (this will be reused)
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"警告：无法加载SentenceTransformer模型: {e}")
    sentence_model = None


class advancedMemAgent:  # 带记忆功能的问答Agent
    def __init__(self, model, backend, retrieve_k, temperature_c5, sglang_host="http://localhost", sglang_port=30000):
        self.memory_system = AgenticMemorySystem(  # 此类用于写入和检索
            model_name='all-MiniLM-L6-v2',
            llm_backend=backend,
            llm_model=model,
            api_key=API_KEY,
            api_base=BASE_URL,
            sglang_host=sglang_host,
            sglang_port=sglang_port
        )
        self.retriever_llm = LLMController(  # 此类负责生成回答
            backend=backend, 
            model=model, 
            api_key=API_KEY,
            api_base=BASE_URL,
            sglang_host=sglang_host, 
            sglang_port=sglang_port
        )
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5  # 对抗问题的采样温度

    def add_memory(self, content, time=None):
        self.memory_system.add_note(content, time=time)

    def retrieve_memory(self, content, k=10):  # 向量检索
        return self.memory_system.find_related_memories_raw(content, k=k)
    
    def retrieve_memory_llm(self, memories_text, query):  # 用llm做二次过滤
        prompt = f"""Given the following conversation memories and a question, select the most relevant parts of the conversation that would help answer the question. Include the date/time if available.

                Conversation memories:
                {memories_text}

                Question: {query}

                Return only the relevant parts of the conversation that would help answer this specific question. Format your response as a JSON object with a "relevant_parts" field containing the selected text.
                And the "relevant_parts" must be a STRING. 
                If no parts are relevant, do not do any things just return the input.

                Example response format:
                {{"relevant_parts": "2024-01-01: Speaker A said something relevant..."}}"""

        response = self.retriever_llm.llm.get_completion(prompt)
        # print("response:{}".format(response))
        return response
    
    def generate_query_llm(self, question):  # 用llm把query扩展成若干关键词
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 
                And the "keywords" field must be a STRING.

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

        response = self.retriever_llm.llm.get_completion(prompt)
        print("response:{}".format(response))
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    # 根据问题生成答案
    def answer_question(self, question: str, category: int, answer: str) -> str:
        # 回答问题
        # 提取关键词
        keywords = self.generate_query_llm(question)
        # 根据关键词检索历史记忆
        raw_context = self.retrieve_memory(keywords,k=self.retrieve_k)
        context = raw_context  # 获取检索到的上下文
        # 不同的种类有不同的prompt
        assert category in [1,2,3,4,5]
        user_prompt = f"""Context:
                {context}

                Question: {question}

                Answer the question based only on the information provided in the context above."""
        temperature = 0.7
        if category == 5:  # adversial question, follow the initial paper.
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('Not mentioned in the conversation')
                answer_tmp.append(answer)
            else:
                answer_tmp.append(answer)
                answer_tmp.append('Not mentioned in the conversation')
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. {question} 
                            
                            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                            """
            temperature = self.temperature_c5
        elif category == 2:
            user_prompt = f"""
                            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

                            Question: {question} Short answer:
                            """
        elif category == 3:
            user_prompt = f"""
                            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            
                            IMPORTANT: Return your answer as a JSON object with this exact format:
                            {{"answer": "your short answer here"}}
                            
                            Data type: "answer" must be a string.
                            Return ONLY the JSON object, no other text.
                            """
        else:
            user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

                            Question: {question} Short answer:
                            """
        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt, temperature=temperature)
        # print(response)
        return response,user_prompt,raw_context


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    # 日志的配置
    logger = logging.getLogger('locomo_eval')  # logger对象名称
    logger.setLevel(logging.INFO)  # 日志级别
    # 日志格式：时间 - 级别 - 消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 控制台处理器，把日志打印到终端
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器，如果提供了log_file路径，则同时写入文件
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # 返回配置好的logger，后续直接logger.info/logger.error(等同于写日志)即可
    return logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None, ratio: float = 1.0, backend: str = "sglang", temperature_c5: float = 0.5, retrieve_k: int = 3, sglang_host: str = "http://localhost", sglang_port: int = 30000):
    # 评估模型在locomo数据集上的表现.
    # 创建带有时间戳的日志文件 “eval_ours_{模型}_{后端}_ratio{采样率}_{时间戳}”
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_ours_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    logger.info(f"加载数据集 {dataset_path}")
    
    # 加载数据集，samples是全部样本
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"加载 {len(samples)} 个样本")
    # 根据比例提取数据集的子集
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"使用 {num_samples} 个样本，是数据集的 ({ratio*100:.1f}% )")
    
    # 储存结果
    results = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    
    # 逐个样本处理
    i = 0
    error_num = 0
    # 缓存记忆文件夹
    memories_dir = os.path.join(os.path.dirname(__file__), "cached_memories_advanced_{}_{}".format(backend, model))
    os.makedirs(memories_dir, exist_ok=True)
    allow_categories = [1,2,3,4,5]
    for sample_idx, sample in enumerate(samples):
        agent = advancedMemAgent(model, backend, retrieve_k, temperature_c5, sglang_host, sglang_port)
        # 创建记忆cache
        memory_cache_file = os.path.join(
            memories_dir,
            f"memory_cache_sample_{sample_idx}.pkl"
        )
        # 检索cache
        retriever_cache_file = os.path.join(
            memories_dir,
            f"retriever_cache_sample_{sample_idx}.pkl"
        )
        # 检索嵌入cache
        retriever_cache_embeddings_file = os.path.join(
            memories_dir,
            f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        if os.path.exists(memory_cache_file):  # 如果某个样本有cache
            logger.info(f"为样本 {sample_idx} 加载缓存的记忆")
            # try:
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            # 给agent添加记忆
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                print(f"找到了检索器的缓存文件:")
                print(f"  - 检索器cache: {retriever_cache_file}")
                print(f"  - 嵌入cache: {retriever_cache_embeddings_file}")
                agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file,retriever_cache_embeddings_file)
            else:
                print(f"没有在 {retriever_cache_file} 找到检索器cache, 从记忆中加载")
                agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(cached_memories, 'all-MiniLM-L6-v2')
            print(agent.memory_system.retriever.corpus)
            logger.info(f"成功加载 {len(cached_memories)} 条记忆")
        else:
            logger.info(f"没有找到样本 {sample_idx} 的缓存记忆. 创建新的记忆.")
            cached_memories = None
            # 处理当前sample
            for _, turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    # 逐句处理，而不是逐轮处理...
                    turn_datatime = turns.date_time
                    conversation_tmp = "Speaker "+ turn.speaker + "says : " + turn.text
                    # 添加记忆，就是add_note
                    agent.add_memory(conversation_tmp,time=turn_datatime)
            # 把记忆存到cache里
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file,retriever_cache_embeddings_file)
            logger.info(f"\n成功缓存 {len(memories_to_cache)} 条记忆")
            
        logger.info(f"\n处理样本 {sample_idx + 1}/{len(samples)}")
        
        for qa in sample.qa:
            if int(qa.category) in allow_categories:
                total_questions += 1
                category_counts[qa.category] += 1
                
                # 生成预测，prediction是模型生成的response
                prediction, user_prompt,raw_context = agent.answer_question(qa.question,qa.category,qa.final_answer)
                try:
                    prediction = json.loads(prediction)["answer"]
                except:
                    prediction = prediction
                    logger.info(f"加载推测结果时失败: {prediction}")
                    error_num += 1
                # 记录日志
                logger.info(f"\n问题 {total_questions}: {qa.question}")
                logger.info(f"推测: {prediction}")
                logger.info(f"正确答案: {qa.final_answer}")
                logger.info(f"用户输入: {user_prompt}")
                logger.info(f"Category: {qa.category}")
                logger.info(f"Raw Context: {raw_context}")
                
                # 计算指标
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0, 
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, 
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }
                # 汇总指标
                all_metrics.append(metrics)
                all_categories.append(qa.category)
                
                # 保存单条结果
                result = {
                    "sample_id": sample_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics
                }
                results.append(result)
                
                # 每10道题打印进度
                if total_questions % 10 == 0:
                    logger.info(f"Processed {total_questions} questions")
        break
    
    # 指标汇总
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    
    # 最终结果
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "individual_results": results
    }
    logger.info(f"错误的答案数量: {error_num}")
    # 保存结果
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"结果保存至 {output_path}")
    
    # 记录summary
    logger.info("\n评估摘要:")
    logger.info(f"所以被评估的问题数量: {total_questions}")
    logger.info("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        logger.info(f"Category {category}: {count} 问题 ({count/total_questions*100:.1f}%)")
    
    logger.info("\n指标汇总:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                logger.info(f"    {stat_name}: {value:.4f}")
    
    return final_results


# 命令行入口
def main():
    parser = argparse.ArgumentParser(description="Evaluate text-only agent on LoComo dataset")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                      help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                      help="OpenAI model to use")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                      help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="sglang",
                      help="Backend to use (openai, ollama, or sglang)")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                      help="Temperature for the model")
    parser.add_argument("--retrieve_k", type=int, default=3,
                      help="Retrieve k")  # k: 10->6->3
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                      help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                      help="SGLang server port (for sglang backend)")
    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # 相对路径变成绝对路径
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    evaluate_dataset(dataset_path, args.model, output_path, args.ratio, args.backend, args.temperature_c5, args.retrieve_k, args.sglang_host, args.sglang_port)


if __name__ == "__main__":
    main()
    '''
    python test_advanced.py --dataset data/locomo10.json --model deepseek-reasoner --backend openai --output A_mem_result.json
    python test_advanced.py --dataset data/locomo10.json --model deepseek-chat --backend openai --output A_mem_result.json
    '''