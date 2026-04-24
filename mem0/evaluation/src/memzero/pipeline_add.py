# 记忆添加模块
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MEM0_DATA_ROOT = os.path.join(PROJECT_ROOT, "mem0_data")

os.environ["MEM0_DIR"] = MEM0_DATA_ROOT
print(f"✅ 已设置 MEM0_DIR: {MEM0_DATA_ROOT}")

from mem0.memory.main import Memory

os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'


custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""

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


# 添加记忆类
class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False):
        self.mem0_client = Memory.from_config(config_dict=config)
        self.batch_size = batch_size  # 批处理大小
        self.data_path = data_path  # 数据集路径
        self.data = None
        self.is_graph = is_graph  # 是否启用知识图谱
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        # 向Mem0添加单条记忆
        for attempt in range(retries):  # 有重试次数
            try:
                _ = self.mem0_client.add(
                    message, user_id=user_id, metadata=metadata)
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        # 处理一个conversation
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"
        # 清空旧记忆
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)
        for key in conversation.keys():  # 遍历当前样本中的所有session
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]
            messages = []
            messages_reverse = []
            for chat in chats:  # 处理session_1
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
            )
            thread_a.start()
            thread_b.start()
            thread_a.join()  # 等待线程结束
            thread_b.join()
        print("成功添加Messages！")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        idx = 9
        item = self.data[idx]
        self.process_conversation(item, idx)
