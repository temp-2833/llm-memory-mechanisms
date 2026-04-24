import json
from collections import deque
try:
    from .utils import get_timestamp, ensure_directory_exists
except ImportError:
    from utils import get_timestamp, ensure_directory_exists


class ShortTermMemory:
    def __init__(self, file_path, max_capacity=10):
        # 初始化
        self.max_capacity = max_capacity
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.memory = deque(maxlen=max_capacity)  # 双端队列
        print(f"🎯 STM初始化 | 容量: {max_capacity} | 文件: {file_path}")
        self.load()

    def add_qa_pair(self, qa_pair):
        # 添加Q&A对
        # Ensure timestamp exists, add if not
        if 'timestamp' not in qa_pair or not qa_pair['timestamp']:
            # 保证有时间戳
            qa_pair["timestamp"] = get_timestamp()
        
        self.memory.append(qa_pair)
        print(f"📥 STM新增QA | 用户输入: {qa_pair.get('user_input','')[:50]}... | "
              f"队列大小: {len(self.memory)} → {len(self.memory)}/{self.max_capacity}")
        # print(f"ShortTermMemory: Added QA. User: {qa_pair.get('user_input','')[:30]}...")
        self.save()

    def get_all(self):
        # 返回当前所有记忆
        return list(self.memory)

    def is_full(self):
        # 检查当前记忆队列是否达到容量上限
        return len(self.memory) >= self.max_capacity

    def pop_oldest(self):
        # 弹出最老的记录
        if self.memory:
            msg = self.memory.popleft()
            print("STM：弹出最老的QA对")
            self.save()
            return msg  # 返回此记录
        return None

    def save(self):
        # 把当前的memory存入文件
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(list(self.memory), f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving ShortTermMemory to {self.file_path}: {e}")

    def load(self):
        # 读取json文件
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure items are loaded correctly, especially if file was empty or malformed
                if isinstance(data, list):
                    # data放进memory
                    self.memory = deque(data, maxlen=self.max_capacity)
                else:
                    self.memory = deque(maxlen=self.max_capacity)
            print(f"STM：从 {self.file_path} 中加载记忆")
        except FileNotFoundError:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            self.memory = deque(maxlen=self.max_capacity)
            print(f"ShortTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.") 