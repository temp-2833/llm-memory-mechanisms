from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import requests
import time
import torch
import faiss


def simple_tokenize(text):
    return word_tokenize(text)


def clean_reasoning_model_output(text):
    if not text:
        return text

    import re
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, api_base: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=6000
            )

            if hasattr(response, 'usage') and response.usage:
                print(
                    f"[Tokens] 输入:{response.usage.prompt_tokens} 输出:{response.usage.completion_tokens} 总计:{response.usage.total_tokens}")

            raw_content = response.choices[0].message.content.strip()
            cleaned_content = clean_reasoning_model_output(raw_content)

            import re
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)

            if json_match:
                return json_match.group(0)
            else:
                print(f"警告：未找到JSON，原始响应: {cleaned_content[:100]}...")  # {cleaned_content[:100]}
                if cleaned_content.startswith('"') or 'keywords' in cleaned_content:
                    return f'{{"raw": "{cleaned_content}"}}'
                else:
                    return '{"keywords": [], "context": "General", "tags": []}'

        except Exception as e:
            print(f"调用API时出现错误: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."

    def chat_completion(self, prompt: str, temperature: float = 0.7):
        messages = [
            {"role": "system", "content": "Act as friend and keep the tone described above in every reply."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            raw_content = response.choices[0].message.content.strip()
            cleaned_content = clean_reasoning_model_output(raw_content)
            return cleaned_content
        except Exception as e:
            print(f"调用API时出现错误: {e}")
            return "错误: 无法从LLM获取回答."

    def get_final_response(self, prompt: str, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": "You must respond with a JSON object."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=6000
            )

            if hasattr(response, 'usage') and response.usage:
                print(
                    f"[Tokens] 输入:{response.usage.prompt_tokens} 输出:{response.usage.completion_tokens} 总计:{response.usage.total_tokens}")

            raw_content = response.choices[0].message.content.strip()

            cleaned_content = clean_reasoning_model_output(raw_content)

            import re
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)

            if json_match:
                return json_match.group(0)
            else:
                print(f"警告：未找到JSON，直接返回原始响应: {cleaned_content[:100]}...")  # {cleaned_content[:100]}
                return cleaned_content

        except Exception as e:
            print(f"调用API时出现错误: {e}")
            # Fallback or error handling
            return "Error: Could not get response from LLM."


# 统一llm工厂
class LLMController:
    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, api_base)
        # elif backend == "ollama":
            # Use LiteLLM to control Ollama with JSON output
            # ollama_model = f"ollama/{model}" if not model.startswith("ollama/") else model
            # self.llm = LiteLLMController(
            #     model=ollama_model,
            #     api_base="http://localhost:11434",
            #     api_key="EMPTY"
            # )
        # elif backend == "sglang":
            # Direct SGLang API calls (better performance, no proxy)
            # self.llm = SGLangController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', or 'sglang'")


class MemoryNote:
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):
        self.content = content

        if llm_controller and any(param is None for param in [keywords, context, category, tags]):

            analysis = self.analyze_content(content, llm_controller)
            print("生成的关键词、上下文和tags：", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)
            
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:            

        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }
            Data type requirements:
            - "keywords": ARRAY of strings, e.g. ["word1", "word2", "word3"]
            - "context": STRING, e.g. "A sentence describing the content"
            - "tags": ARRAY of strings, e.g. ["category1", "category2", "category3"]
            
            Content for analysis: """ + str(content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"analyze_content函数中出现JSON解析错误: {e}")
                print(f"原始response: {response}")
                analysis = {
                    "keywords": [],
                    "context": "General",
                    "tags": []
                }
            return analysis
        except Exception as e:
            print(f"分析内容analyzing content出现错误: {str(e)}")
            # print(f"Raw response: {response}")
            return {
                "keywords": [],
                "context": "General",
                "category": "Uncategorized",
                "tags": []
            }


class SimpleEmbeddingRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.index = None
        self.dimension = None
        self.document_ids = {}
        
    def add_documents(self, documents: List[str]):

        if not documents:
            return
        if not self.corpus:
            self.corpus = documents
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

        embeddings = self.model.encode(documents)

        if self.index is None:

            self.dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(self.dimension)

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: str, k: int = 3) -> List[int]:

        if not self.corpus:
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        if k > len(self.corpus):
            k = len(self.corpus)
        distances, indices = self.index.search(query_embedding, k)
        return indices[0].tolist()
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):

        if self.index is not None:
            faiss.write_index(self.index, retriever_cache_embeddings_file)

        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'dimension': self.dimension
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):

        print(f"加载检索器 from {retriever_cache_file} 和 {retriever_cache_embeddings_file}")
        

        if os.path.exists(retriever_cache_embeddings_file):
            print(f"加载 embeddings from {retriever_cache_embeddings_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                self.dimension = state.get('dimension')
                print(f"加载好的 corpus 有 {len(self.corpus)} 个 documents")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")

        if os.path.exists(retriever_cache_file):
            print(f"加载 corpus from {retriever_cache_file}")
            self.index = faiss.read_index(retriever_cache_embeddings_file)
            print(f"FAISS索引大小: {self.index.ntotal}")
        else:
            print(f"FAISS index file not found: {retriever_cache_embeddings_file}")
            # 如果文件不存在，重新从corpus构建索引
            if self.corpus:
                print("重新构建FAISS索引...")
                self.rebuild_faiss_index()
        return self

    def rebuild_faiss_index(self):
        if not self.corpus:
            return

        embeddings = self.model.encode(self.corpus)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"重建FAISS索引完成，包含 {self.index.ntotal} 个向量")

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':

        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever


def clean_memory_data(data):

    if "tags_to_update" in data:
        data["tags_to_update"] = [str(tag) for tag in data["tags_to_update"] if tag]

    if "new_tags_neighborhood" in data:

        cleaned = []
        for neighbor_tags in data["new_tags_neighborhood"]:
            if isinstance(neighbor_tags, list):
                cleaned.append([str(tag) for tag in neighbor_tags if tag])
            else:
                cleaned.append([str(neighbor_tags)])
        data["new_tags_neighborhood"] = cleaned
    return data


class AgenticMemorySystem:

    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,  # 记忆演化阈值
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        self.memories = {}  # id -> MemoryNote
        self.retriever = SimpleEmbeddingRetriever(model_name)
        # 初始化OpenAIController类
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base, sglang_host, sglang_port)
        # self.evolution_system_prompt = '''
        #                         You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
        #                         Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
        #                         Make decisions about its evolution.
        #
        #                         The new memory context:
        #                         {context}
        #                         content: {content}
        #                         keywords: {keywords}
        #
        #                         The nearest neighbors memories:
        #                         {nearest_neighbors_memories}
        #
        #                         Based on this information, determine:
        #                         1. Should this memory be evolved? Consider its relationships with other memories.
        #                         2. What specific actions should be taken (strengthen, update_neighbor)?
        #                            2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
        #                            2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
        #                         Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
        #                         Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
        #                         The number of neighbors is {neighbor_number}.
        #                         Return your decision in JSON format with the following structure:
        #                         {{
        #                             "should_evolve": True or False,
        #                             "actions": ["strengthen", "update_neighbor"],
        #                             "suggested_connections": ["neighbor_memory_ids"],
        #                             "tags_to_update": ["tag_1",..."tag_n"],
        #                             "new_context_neighborhood": ["new context",...,"new context"],
        #                             "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
        #                         }}
        #
        #                         Data type requirements (must match exactly):
        #                         - "suggested_connections": array of integers
        #                         - "new_tags_neighborhood": array of arrays of strings
        #                         '''
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                                           
                                CRITICAL CONSTRAINTS - STRICTLY ENFORCED:
                                - Each new_context MUST be under 40 words (STRICT LIMIT)
                                - Each tags list MUST NOT exceed 8 tags (HARD LIMIT)
                                - ABSOLUTELY NO phrases like "now understood as" (PROHIBITED)
                                - VIOLATIONS will cause system errors
                                
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}

                                Data type requirements (must match exactly):
                                - "suggested_connections": array of integers
                                - "new_tags_neighborhood": array of arrays of strings
                                '''
        self.evo_cnt = 0  # evolve计数器
        self.evo_threshold = evo_threshold  # 演变阈值

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        # 添加新的记忆note
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)

        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note  # 存储记忆
        # 给检索器添加文档
        self.retriever.add_documents(["content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)])
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()  # 定期巩固记忆
        return note.id
    
    def consolidate_memories(self):
        """巩固记忆：用新文档【更新】检索器
        此函数重新初始化检索器并用所有记忆文档（包括它们的上下文、关键词和标签）更新它，
        以确保检索系统拥有所有记忆的最新状态。
        """
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'  # 默认
        # 重置检索器
        self.retriever = SimpleEmbeddingRetriever(model_name)
        # 重新添加所有记忆文档(包括元数据)
        for memory in self.memories.values():
            # 把记忆metadata合并成一个可搜索文档
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def process_memory(self, note: MemoryNote) -> bool:

        neighbor_memory, indices = self.find_related_memories(note.content, k=5)  # 5—>3->5
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        print("prompt_memory的内容", prompt_memory)
        if len(indices) == 0:
            response = json.dumps({
                                  "should_evolve": False,
                                  "actions": [],
                                  "suggested_connections": [],
                                  "tags_to_update": [],
                                  "new_context_neighborhood": [],
                                  "new_tags_neighborhood": []
                                })
        else:
            response = self.llm_controller.llm.get_completion(prompt_memory)
        try:
            print("LLM的回答", response, type(response))
            response_json = json.loads(response)
            response_json = clean_memory_data(response_json)
            print("LLM的JSON版回答", response_json, type(response_json))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            return False, note

        should_evolve = response_json["should_evolve"]
        if should_evolve:  # 需要演变
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":

                    suggest_connections = response_json["suggested_connections"]
                    # 当前记忆应该更新成什么标签["标签1", "标签2"...]
                    new_tags = response_json["tags_to_update"]
                    # 把建议的连接添加到当前记忆的link中
                    note.links.extend(suggest_connections)
                    # 更新标签
                    note.tags = new_tags
                elif action == "update_neighbor":
                    # 更新邻居
                    # 邻居们的新上下文(新的解读)列表["新上下文1", "新上下文2"...]
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    # 邻居们的新标签列表
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    # 获取所有记忆的列表和ID列表，用于通过索引查找
                    noteslist = list(self.memories.values())  # [MemoryNote1, MemoryNote2...]
                    notes_id = list(self.memories.keys())  # [id1, id2...]
                    print("找到的邻居记忆的索引：", indices)

                    # 逐个更新邻居记忆
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # 获取邻居新的tags
                        tag = new_tags_neighborhood[i]
                        # 获取邻居新的上下文
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        # 邻居在noteslist中的索引
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # 更新标签和上下文
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve,note

    def find_related_memories(self, query: str, k: int = 3) -> List[MemoryNote]:  # 5->3
        # 使用混合检索查找相关记忆
        if not self.memories:
            return "",[]
        # 获取相关记忆的索引
        indices = self.retriever.search(query, k)
        # 转换成list
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        for i in indices:  # 变成字符串
            memory_str += ("memory index:" + str(i) + "\t talk start time:" +
                           all_memories[i].timestamp + "\t memory content: " +
                           all_memories[i].content + "\t memory context: " +
                           all_memories[i].context + "\t memory keywords: " +
                           str(all_memories[i].keywords) + "\t memory tags: " +
                           str(all_memories[i].tags) + "\n")
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 3) -> List[MemoryNote]:  # 5->3
        # 使用混合检索查找相关记忆（返回原始格式字符串）
        if not self.memories:
            return []
        # 获取相关记忆索引
        indices = self.retriever.search(query, k)
        # 转换成list
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        for i in indices:
            memory_str += ("talk start time:" + all_memories[i].timestamp +
                           "memory content: " + all_memories[i].content +
                           "memory context: " + all_memories[i].context +
                           "memory keywords: " + str(all_memories[i].keywords)
                           + "memory tags: " + str(all_memories[i].tags) + "\n")
            # neighborhood = all_memories[i].links
            # for neighbor in neighborhood:
            #     memory_str += ("talk start time:" + all_memories[neighbor].timestamp +
            #                    "memory content: " + all_memories[neighbor].content +
            #                    "memory context: " + all_memories[neighbor].context +
            #                    "memory keywords: " + str(all_memories[neighbor].keywords) +
            #                    "memory tags: " + str(all_memories[neighbor].tags) + "\n")
            #     if j >=k:
            #         break
            #     j += 1
        return memory_str


# 运行整个系统
def run_tests():
    # 运行系统测试
    print("开始记忆系统测试...")
    # 使用OpenAI后端初始化记忆系统
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',
        llm_backend='openai',
        llm_model='gpt-4o-mini'
    )
    print("\n添加测试记忆...")

    memory_ids = []
    memory_ids.append(memory_system.add_note(
        "Neural networks are composed of layers of neurons that process information."
    ))
    
    memory_ids.append(memory_system.add_note(
        "Data preprocessing involves cleaning and transforming raw data for model training."
    ))
    
    print("\n询问相关记忆...")
    query = MemoryNote(
        content="How do neural networks process data?",
        llm_controller=memory_system.llm_controller
    )
    
    related = memory_system.find_related_memories(query.content, k=2)
    print("找到的相关记忆：", related)
    print("\n最终结果:")
    for i, memory in enumerate(related, 1):
        print(f"\n{i}. Memory:")
        print(f"内容: {memory.content}")
        print(f"Category: {memory.category}")
        print(f"关键词: {memory.keywords}")
        print(f"Tags: {memory.tags}")
        print(f"上下文: {memory.context}")
        print("-" * 50)


if __name__ == "__main__":
    run_tests()