# 负责从已存储的记忆中检索信息并生成答案
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from ..prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

# --------------------------------------
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MEM0_DATA_ROOT = os.path.join(PROJECT_ROOT, "mem0_data")

os.environ["MEM0_DIR"] = MEM0_DATA_ROOT
print(f"✅ 已设置 MEM0_DIR: {MEM0_DATA_ROOT}")
from mem0.memory.main import Memory

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'

config = {
    "llm": {
        "provider": "",
        "config": {
            "model": "",
            "api_key": "",
            "base_url": "",
            "temperature": 0.1,
            "max_tokens": 4000,
        },
    },
    "vector_store": {
        "provider": "faiss",
        "config": {
            "collection_name": "locomo_collection",
            "path": os.path.join(MEM0_DATA_ROOT, "locomo_faiss_index"),
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "huggingface_base_url": None,
        },
    },
}

class MemorySearch:
    def __init__(self, output_path="results.json", top_k=6, filter_memories=False, is_graph=False):

        self.mem0_client = Memory.from_config(config_dict=config)
        self.top_k = top_k  # 搜索的记忆数量
        self.openai_client = OpenAI(api_key="sk-abba92a8ee1641c59fdcb83f5ce388de", base_url="https://api.deepseek.com")
        # self.results = defaultdict(list)
        self.results = []
        self.output_path = output_path  # 结果输出路径
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("含有知识图谱的检索")
                    # 调用Memory类里的search函数
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        limit=self.top_k,
                        rerank=False
                    )
                    memories = memories.get("results", [])
                else:
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        limit=self.top_k,
                        rerank=False
                    )
                    memories = memories.get("results", [])
                    print("搜索到的相关记忆如下：")
                    print(memories)
                break
            except Exception as e:
                print("重试...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:

            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            # 图谱搜索返回格式
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
               # 记忆、时间和分数
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):

        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )
        # 格式化记忆文本，用于插入prompt
        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        # 回答的prompt
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )
        # 调用API
        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model="deepseek-reasoner", messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1  # 计算应答时间
        return (
            response.choices[0].message.content,  # llm生成的回答
            speaker_1_memories,  # 相关记忆
            speaker_2_memories,
            speaker_1_memory_time,  # 记忆时间
            speaker_2_memory_time,
            speaker_1_graph_memories,  # 如果没有启用图，则是None
            speaker_2_graph_memories,
            response_time,  # llm应答时间
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        # 处理单个问题
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")
        if category == 5:
            print(f"跳过 category=5 的问题: {question[:50]}...")
            # 返回一个标记为跳过的结果
            return {
                "sample_id": "sample_id",
                "speaker_a": "speaker_a",
                "speaker_b": "speaker_b",
                "question": question,
                "system_answer": "SKIPPED - Category 5",  # 标记为跳过
                "original_answer": answer,
                "category": category,
                "evidence": evidence,
                "response_time": 0
            }

        # 调用上面的回答问题函数
        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "sample_id": "sample_id",
            "speaker_a": "speaker_a",
            "speaker_b": "speaker_b",
            "question": question,
            "system_answer": response,
            "original_answer": answer,
            "category": category,
            "evidence": evidence,
            "response_time": response_time,
        }


        return result

    def process_data_file(self, file_path):
        # 处理整个数据集文件
        with open(file_path, "r") as f:
            data = json.load(f)
        target_idx = 9  # 要处理的样本索引
        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            if idx != target_idx:
                continue  # 跳过其他样本
            qa = item["qa"]  # 一个样本的所有qa
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
            ):

                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                # self.results[idx].append(result)
                self.results.append(result)

            # 处理一个样本就结束
            break

        # 最后保存所有的
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

