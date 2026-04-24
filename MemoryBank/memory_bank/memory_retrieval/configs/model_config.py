import torch.cuda
import torch.backends
import os

# 嵌入模型字典，存储各类文本嵌入模型及其对应的HuggingFace模型路径
embedding_model_dict = {
    # 中文模型
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    # 英文模型
    'multilingual-mpnet':"paraphrase-multilingual-mpnet-base-v2",
    'mpnet':"all-mpnet-base-v2",
    'minilm-l6':'all-MiniLM-L6-v2',
    'minilm-l12':'all-MiniLM-L12-v2',
    'multi-qa':"multi-qa-mpnet-base-dot-v1",
    'alephbert':'imvladikon/sentence-transformers-alephbert',
    'sbert-cn':'uer/sbert-base-chinese-nli'
} 

EMBEDDING_MODEL_CN = "text2vec"  # 中文默认
EMBEDDING_MODEL_EN = "minilm-l6"  # 英文默认

# 嵌入模型运行的设备
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 支持的LLM
llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
}

LLM_MODEL = "chatglm-6b"  # 默认

USE_PTUNING_V2 = False  # 是否使用P-tuning-v2前缀编码器进行微调

# LLM运行设备
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 向量存储根路径
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")
# 上传文件根路径
UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "")

# 匹配后单段上下文长度
CHUNK_SIZE = 200