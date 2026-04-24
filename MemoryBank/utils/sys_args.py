from transformers import HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    # 相关参数
    memory_search_top_k: int = field(default=2)  # 每次检索几段对话
    memory_basic_dir: str = field(default='../../memories/')  # 记忆文件的存储根目录
    memory_file: str = field(default='update_memory_0512_eng.json')  # 记忆文件
    language: str = field(default='en')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    # 是否启用遗忘机制
    enable_forget_mechanism: bool = field(default=False)


data_args = HfArgumentParser(DataArguments).parse_args_into_dataclasses()[0]