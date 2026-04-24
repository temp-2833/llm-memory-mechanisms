import json, datetime
import random, copy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from typing import List, Tuple, Optional
from langchain.vectorstores import FAISS
from textsplitter import ChineseTextSplitter
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from configs.model_config import *

# 向量检索时返回的最相关文本块数量
VECTOR_SEARCH_TOP_K = 6
# LLM输入的历史对话长度
LLM_HISTORY_LEN = 3
import math


def forgetting_curve(t, S):
    return math.exp(-t / 5*S)


class MemoryForgetterLoader(UnstructuredFileLoader):
    # 记忆管理和遗忘加载器
    def __init__(self, filepath,language,mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.memory_level = 3
        self.total_date = 30
        self.language = language
        self.memory_bank = {}   # 存储所有用户记忆的字典

    def _get_date_difference(self, date1: str, date2: str) -> int:
        # 计算两个日期之间的天数差
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1, date_format)
        d2 = datetime.datetime.strptime(date2, date_format)
        return (d2 - d1).days

    def update_memory_when_searched(self, recalled_memos,user,cur_date):
        # 当记忆被检索时，更新记忆强度
        for recalled in recalled_memos:
            recalled_id = recalled.metadata['memory_id']  # 获取id
            recalled_date = recalled_id.split('_')[1]
            # 在用户的记忆历史中查找并更新对应的记忆项
            for i,memory in enumerate(self.memory_bank[user]['history'][recalled_date]):
                if memory['memory_id'] == recalled_id:
                    self.memory_bank[user]['history'][recalled_date][i]['memory_strength'] += 1
                    self.memory_bank[user]['history'][recalled_date][i]['last_recall_date'] = cur_date
                    break
    
    def write_memories(self, out_file):
        # 把记忆写入JSON文件
        with open(out_file, "w", encoding="utf-8") as f:
            print(f'Successfully write to {out_file}')
            json.dump(self.memory_bank, f, ensure_ascii=False, indent=4)
    
    def load_memories(self, memory_file):
        # 从JSON文件加载记忆数据
        # print(memory_file)
        with open(memory_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)

    def initial_load_forget_and_save(self,name,now_date):
        # 初始化、应用遗忘曲线并保存记忆
        # 核心方法：加载记忆->应用遗忘算法->保存更新后的记忆
        docs = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            memories = json.load(f)
            # 遍历所有用户的记忆
            for user_name, user_memory in memories.items():
                # if user_name != name:
                #     continue
                if 'history' not in user_memory.keys():
                    continue
                self.memory_bank[user_name] = copy.deepcopy(user_memory)
                # 遍历用户的每日对话记录
                for date, content in user_memory['history'].items():
                    # 根据语言设置记忆字符串的前缀
                     memory_str = f'时间{date}的对话内容：' if self.language=='cn' else f'Conversation content on {date}:'
                     user_kw = '[|用户|]：' if self.language=='cn' else '[|User|]:'
                     ai_kw = '[|AI伴侣|]：' if self.language=='cn' else '[|AI|]:'
                    # 需要遗忘的记忆索引
                     forget_ids = []
                    # 处理每日的每个对话
                     for i,dialog in enumerate(content):
                        tmp_str = memory_str
                        if not isinstance(dialog,dict):
                            dialog = {'query':dialog[0],'response':dialog[1]}
                            self.memory_bank[user_name]['history'][date][i] = dialog
                        # 提取对话内容
                        query = dialog['query']
                        response = dialog['response']
                        memory_strength = dialog.get('memory_strength',1)
                        last_recall_date = dialog.get('last_recall_date',date)
                        memory_id = dialog.get('memory_id',f'{user_name}_{date}_{i}')
                        # 构建记忆文本
                        tmp_str += f'{user_kw} {query.strip()}; '
                        tmp_str += f'{ai_kw} {response.strip()}'
                        metadata = {'memory_strength':memory_strength,
                                    'memory_id':memory_id,'last_recall_date':last_recall_date,"source": memory_id}
                        # 更新内存中的记忆数据
                        self.memory_bank[user_name]['history'][date][i].update(metadata)
                        # 应用遗忘曲线算法
                        days_diff = self._get_date_difference(last_recall_date, now_date)
                        retention_probability = forgetting_curve(days_diff,memory_strength)
                        print(days_diff,memory_strength,retention_probability)
                        # 随机决定是否遗忘：保留概率为retention_probability的记忆
                        if random.random() > retention_probability:
                            forget_ids.append(i)
                        else:
                            docs.append(Document(page_content=tmp_str,metadata=metadata))
                    # 处理需要遗忘的记忆(从后往前删除)
                     print(user_name,date,forget_ids)
                     if len(forget_ids) > 0:
                         forget_ids.sort(reverse=True)
                         for idd in forget_ids:
                             self.memory_bank[user_name]['history'][date].pop(idd)
                     if len(self.memory_bank[user_name]['history'][date])==0:
                            self.memory_bank[user_name]['history'].pop(date)
                            self.memory_bank[user_name]['summary'].pop(date)
                     if 'summary' in self.memory_bank[user_name].keys():
                        if date in self.memory_bank[user_name]['summary'].keys():
                            if not isinstance(self.memory_bank[user_name]["summary"][date],dict):
                                self.memory_bank[user_name]["summary"][date] = {'content':self.memory_bank[user_name]["summary"][date]}
                            summary_str = self.memory_bank[user_name]["summary"][date]["content"] if isinstance(self.memory_bank[user_name]["summary"][date],dict) else self.memory_bank[user_name]["summary"][date]
                            summary = f'时间{date}的对话总结为：{summary_str}' if self.language=='cn' else f'The summary of the conversation on {date} is: {summary_str}'
                            memory_strength = self.memory_bank[user_name]['summary'][date].get('memory_strength',1) 
                            last_recall_date = self.memory_bank[user_name]["summary"][date].get('last_recall_date',date) 
                            metadata = {'memory_strength':memory_strength,
                                    'memory_id':f'{user_name}_{date}_summary','last_recall_date':last_recall_date,"source":f'{user_name}_{date}_summary'}
                            if isinstance(self.memory_bank[user_name]["summary"][date],dict):    
                                self.memory_bank[user_name]['summary'][date].update(metadata)
                            else:
                                self.memory_bank[user_name]['summary'][date] = {'content':self.memory_bank[user_name]['summary'][date],**metadata}
                            docs.append(Document(page_content=summary,metadata=metadata))
        self.write_memories(self.filepath) 
        return docs

    
def get_docs_with_score(docs_with_score):
    # 把(文档，分数)元组列表转换为文档列表，并将分数添加到文档元数据中
    docs=[]
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    # 把连续整数列表分割成连续子序列
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


def load_memory_file(filepath,user_name,cur_date='',language='cn'):
    memory_loader = MemoryForgetterLoader(filepath,language)
    # docs = loader.load(user_name)
    textsplitter = ChineseTextSplitter(pdf=False)
    # if not os.path.exists(filepath.replace('.json','_forget_format.json')):
    print('write to ',os.path.exists(filepath.replace('.json','_forget_format.json')))
    if not cur_date:
        cur_date = datetime.date.today().strftime("%Y-%m-%d")
     
    # if not os.path.exists(os.path.exists(filepath.replace('.json','_forget_format.json'))):
    docs = memory_loader.initial_load_forget_and_save(user_name,cur_date)
    docs = textsplitter.split_documents(docs)
    
    return docs, memory_loader 
    

def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
    # FAISS相似度搜索的扩展方法，在其基础上，合并属于同一记忆的相邻文档块
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        for j, i in enumerate(indices[0]):
            if i == -1:
                # 返回-1表示没有足够的文档
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
    # 本地记忆检索系统(核心)，负责记忆的向量化、存储、检索和管理
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL_CN,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model])
        self.user = ''
        self.top_k = top_k
        self.memory_loader = None
        self.memory_path = ''
    
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date:str=None):
        # 初始化记忆向量存储
        loaded_files = []
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        self.user=user_name
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                # try:
                self.memory_path = filepath
                docs,memory_loader = load_memory_file(filepath,user_name,cur_date,self.language)
                print(f"{file} 已成功加载")
                loaded_files.append(filepath)
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    # if user_name not in fullfilepath:
                    #     continue
                    try:
                        self.memory_path = fullfilepath
                        new_docs, memory_loader = load_memory_file(fullfilepath,user_name,cur_date,self.language)
                        docs += new_docs
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    self.memory_path = file
                    new_docs,memory_loader = load_memory_file(file,user_name,cur_date,self.language)
                    docs += new_docs
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        self.memory_loader = memory_loader
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f'Saving to {vs_path}')
            vector_store.save_local(vs_path)
            return vs_path, loaded_files
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
                    vector_store,
                    cur_date=''):
        # 搜索记忆
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ''
        date_docs = []
        dates = []
        cur_date = cur_date if cur_date else datetime.date.today().strftime("%Y-%m-%d") 
        for doc in related_docs:
            #saving updated memory
            
            doc.page_content = doc.page_content.replace(f'时间{doc.metadata["source"]}的对话内容：','').strip()
            if doc.metadata["source"] != pre_date:
                # date_docs.append(f'在时间{doc.metadata["source"]}的回忆内容是：{doc.page_content}')
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}' 
        self.memory_loader.update_memory_when_searched(related_docs,user=self.user,cur_date=cur_date)
        self.save_updated_memory()
        # memory_contents = [doc.page_content for doc in related_docs]
        # memory_contents = [f'在时间'+doc.metadata['source']+'的回忆内容是：'+doc.page_content for doc in related_docs]
        return date_docs, ', '.join(dates) 
    
    def save_updated_memory(self):
        # 保存更新后的记忆到文件
        self.memory_loader.write_memories(self.memory_path)#.replace('.json','_forget_format.json'))
    