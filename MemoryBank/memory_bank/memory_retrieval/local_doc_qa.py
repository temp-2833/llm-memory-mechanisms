from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter, NLTKTextSplitter

import datetime
from typing import List, Tuple
from langchain.docstore.document import Document
import numpy as np
import json
import os
from pathlib import Path

# 向量存储返回的最相关的文本块数量
VECTOR_SEARCH_TOP_K = 3
# 匹配后单段上下文长度
CHUNK_SIZE = 200
EMBEDDING_DEVICE = "cpu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# 向量存储根路径
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")


class JsonMemoryLoader(UnstructuredFileLoader):
    # JSON格式记忆加载器
    def __init__(self, filepath,language,mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.language = language

    def _get_metadata(self, date: str) -> dict:  # 生成元数据
        return {"source": date}
    
    def load(self,name):
        # 加载指定用户的记忆数据
        user_memories = []
        print(self.file_path)
        f = open(self.filepath, "r", encoding="utf-8")
        memories = json.loads(f.read())
        #print(memories)
        for user_name, user_memory in memories.items():
            if user_name != name:
                continue
            #print(user_memory)
            user_memories = []
            if 'history' not in user_memory.keys():
                continue
            for date, content in user_memory['history'].items():
                metadata = self._get_metadata(date)
                memory_str = f'时间{date}的对话内容：' if self.language=='cn' else f'Conversation content on {date}:'
                user_kw = '[|用户|]：' if self.language=='cn' else '[|User|]:'
                ai_kw = '[|AI恋人|]：' if self.language=='cn' else '[|AI|]:'
                print("content:")
                print(content)
                for i,(dialog) in enumerate(content):
                    query, response = dialog['query'],dialog['response']

                    tmp_str = memory_str
                    tmp_str += f'{user_kw} {query.strip()}; '
                    tmp_str += f'{ai_kw} {response.strip()}'
                    user_memories.append(Document(page_content=tmp_str, metadata=metadata)) 
                # memory_str += '\n'
                if 'summary' in user_memory.keys():
                    if date in user_memory['summary'].keys():
                        print("处理了summary相关")  # 没有走进来
                        summary = f'时间{date}的对话总结为：{user_memory["summary"][date]}' if self.language=='cn' else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                        # memory_str += summary
                        user_memories.append(Document(page_content=summary, metadata=metadata))
        f.close() 
        return user_memories
     
    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None,name=''
    ) -> List[Document]:
        # 加载文档并进行文本分割
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load(name)
        results = _text_splitter.split_documents(docs)

        return results


def load_memory_file(filepath,user_name,language='cn'):
    # 加载记忆文件的便捷函数
    loader = JsonMemoryLoader(filepath,language)
    docs = loader.load(user_name)
    # docs = loader.load_and_split(
    #     text_splitter=RecursiveCharacterTextSplitter(
    #         chunk_size=200,
    #         chunk_overlap=20
    #     ),
    #     name=user_name
    # )
    return docs
 

def get_docs_with_score(docs_with_score):
    # 将(文档，分数)元组列表转换成文档列表，并将分数添加到元数据中
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    # 把连续整数列表分割为连续子序列
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
    # FAISS向量相似度搜索的自定义扩展方法
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # 没有足够的结果
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, len(docs)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        # print(doc0.metadata)
                        # exit()
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        # print(doc0)
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, scores[0][j]))
        return docs


class LocalMemoryRetrieval:
    # 本地记忆检索系统
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        # 初始化
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.top_k = top_k
    
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date: str = None):

        loaded_files = []
        filepath = Path(filepath)
        if not filepath.exists():
            print("路径不存在")
            return None, None
        elif filepath.is_file():
            docs = load_memory_file(str(filepath),user_name,self.language)
            print(f"{filepath.name} 已成功加载")
            loaded_files.append(filepath)
        elif filepath.is_dir():
            docs = []
            for child in filepath.iterdir():
                try:
                    docs += load_memory_file(str(child),user_name,self.language)
                    print(f"{child.name} 已成功加载")
                    loaded_files.append(str(child))
                except Exception as e:
                    print(e)
                    print(f"{child.name} 未能成功加载")

        if len(docs) > 0:
            vs_path = Path(vs_path) if vs_path else None
            if vs_path and vs_path.is_dir():
                vector_store = FAISS.load_local(str(vs_path), self.embeddings)
                print(f'从之前的记忆索引位置 {vs_path} 加载.')
                vector_store.add_documents(docs)
            else:
                # 新建
                stem = filepath.stem
                if not vs_path:
                    # 时间戳
                    time_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 建文件
                    vs_path = Path(VS_ROOT_PATH)/f"{stem}_FAISS_{time_suffix}"
                    vs_path.mkdir(parents=True, exist_ok=True)
                vector_store = FAISS.from_documents(docs, self.embeddings)
            vector_store.save_local(str(vs_path))
            return str(vs_path), loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files
    
    def load_memory_index(self,vs_path):
        # 加载已保存的记忆向量索引
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size=self.chunk_size
        return vector_store
    
    def search_memory(self,
                    query,
                    vector_store):

        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ''
        date_docs = []
        dates = []
        for doc in related_docs:
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}'
        return date_docs, ', '.join(dates) 
    

