import json, datetime
import random, copy
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from typing import List, Tuple, Optional
from langchain.vectorstores import FAISS
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
import os

# 向量检索时返回的最相关文本块数量
VECTOR_SEARCH_TOP_K = 6
CHUNK_SIZE = 200
EMBEDDING_DEVICE = "cpu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# LLM输入的历史对话长度
LLM_HISTORY_LEN = 3
# 向量存储根路径
VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")
import math
from pathlib import Path


# def forgetting_curve_calibrated(t, S, calibration_factor=None):
#     """
#     为locomo数据集校准的遗忘曲线
#     Args:
#         t: 天数差（原始可能很大，如1000天）
#         S: 记忆强度（1.0为基础）
#         calibration_factor: 校准因子，越大遗忘越慢
#     Returns:
#         保留概率 (0-1)
#     """
#     if calibration_factor is None:
#         # 默认校准因子，使平均保留率≈0.33
#         # 对于t≈1000天，S=1.0时，原版exp(-1000/5)=e^{-200}≈0
#         # 校准后exp(-1000/(5*200))=e^{-1}≈0.37
#         calibration_factor = 100  # 200->100
#     # 应用校准因子：t / (5 * S * calibration_factor)
#     return math.exp(-t / (5 * S * calibration_factor))

def forgetting_curve_calibrated(t, S, calibration_factor=None):
    # 基于遗忘曲线计算在时间t时的信息保留程度，t是自信息学习以来经过的时间(天)
    # S是记忆强度(越大，记忆遗忘越慢)，返回值是时间t时的信息保留率(0-1之间)
    return math.exp(-t / 5*S)

def extract_date_from_timestamp(timestamp):
    """从locomo时间戳中提取日期"""
    if not timestamp:
        return "unknown_date"
    try:
        import re
        # 匹配 "on 18 March, 2022" 或 "18 March, 2022"
        date_pattern = r'(?:on )?(\d{1,2} \w+, \d{4})'
        match = re.search(date_pattern, timestamp)
        if match:
            return match.group(1)
    except:
        pass
    return timestamp


class MemoryForgetterLoader(UnstructuredFileLoader):
    # 记忆管理和遗忘加载器
    def __init__(self, filepath, language, mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.memory_level = 3
        self.total_date = 30
        self.language = language
        self.memory_bank = {}  # 存储所有用户记忆的字典

        # 遗忘曲线校准参数
        self.calibration_factor = 100  # 默认值，可通过实验调整 200->100
        self.target_retention = 0.33  # 目标平均保留率

    def _get_date_difference(self, date1: str, date2: str) -> int:
        """计算两个日期之间的天数差"""
        def parse_date(date_str):
            """解析各种日期格式"""
            if not date_str:
                return datetime.datetime.now()
            # 1. 标准格式
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
                try:
                    return datetime.datetime.strptime(date_str, fmt)
                except:
                    continue
            # 2. locomo格式："6:59 pm on 18 March, 2022"
            import re
            # 匹配 "on 18 March, 2022" 或 "18 March, 2022"
            match = re.search(r'(?:on )?(\d{1,2}) (\w+), (\d{4})', date_str)
            if match:
                day, month_str, year = match.groups()
                month_map = {
                    'January': 1, 'February': 2, 'March': 3, 'April': 4,
                    'May': 5, 'June': 6, 'July': 7, 'August': 8,
                    'September': 9, 'October': 10, 'November': 11, 'December': 12,
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                return datetime.datetime(int(year), month_map.get(month_str, 1), int(day))
            # 3. 实在不行，用今天
            print(f"警告：无法解析日期 '{date_str}'，使用今天")
            return datetime.datetime.now()
        d1 = parse_date(date1)
        d2 = parse_date(date2)
        print(f"d1:{d1}  d2:{d2}")
        return abs((d2 - d1).days)

    def update_memory_when_searched(self, recalled_memos, user, cur_date):
        """适配locomo的更新逻辑"""
        for recalled in recalled_memos:
            recalled_id = recalled.metadata['memory_id']  # 获取id
            # 解析ID：sample_id_date_index
            parts = recalled_id.split('_')
            if len(parts) >= 3:
                sample_id = '_'.join(parts[:-2])  # 处理可能有下划线的sample_id
                date = parts[-2]
                index = int(parts[-1])

                # 在memory_bank中查找并更新
                if (sample_id in self.memory_bank and
                        'history' in self.memory_bank[sample_id] and
                        date in self.memory_bank[sample_id]['history'] and
                        index < len(self.memory_bank[sample_id]['history'][date])):

                    memory = self.memory_bank[sample_id]['history'][date][index]
                    if memory.get('memory_id') == recalled_id:
                        # 更新记忆强度（原项目是+1，我们也可以调整）
                        memory['memory_strength'] = min(memory.get('memory_strength', 1.0) + 0.1, 3.0)
                        memory['last_recall_date'] = cur_date
                        print(f"更新记忆 {recalled_id}: 强度 → {memory['memory_strength']:.2f}")

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


    def initial_load_forget_and_save(self, process_index, now_date, calibration_factor=None):
        """
        为locomo数据集适配的初始加载、遗忘和保存
        Args:
            process_index: 样本索引
            now_date: 当前日期（如"2026-03-13"）
            calibration_factor: 可选，覆盖默认校准因子
        """
        if calibration_factor is not None:
            self.calibration_factor = calibration_factor

        docs = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sample = dataset[process_index]
        sample_id = sample.get('sample_id', f'user_{process_index}')
        print(f"正在处理样本{process_index + 1}/{len(dataset)}: {sample_id}")
        print(f"使用校准因子: {self.calibration_factor}")

        # 初始化memory_bank结构
        self.memory_bank[sample_id] = {
            "history": {},
            "summary": {}
        }
        conversation_data = sample["conversation"]
        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]
        # 获取所有session_keys
        session_keys = [key for key in conversation_data.keys()
                        if key.startswith("session_") and not key.endswith("_date_time")]
        # 第一步：按日期分组对话（仿照原项目结构）
        date_to_dialogs = {}
        for session_key in session_keys:
            timestamp_key = f"{session_key}_date_time"
            timestamp = conversation_data.get(timestamp_key, "")
            # 使用 extract_date_from_timestamp 提取日期
            date = extract_date_from_timestamp(timestamp)
            if date not in date_to_dialogs:
                date_to_dialogs[date] = []
            # 处理session内的对话配对
            dialogs_in_session = conversation_data[session_key]
            current_user_input = ""
            for dialog in dialogs_in_session:
                speaker = dialog["speaker"]
                text = dialog["text"]
                if "blip_caption" in dialog and dialog["blip_caption"]:
                    text = f"{text} (image description: {dialog['blip_caption']})"
                if speaker == speaker_a:
                    # 用户发言，开始新的对话对
                    if current_user_input:  # 保存前一个未配对的用户发言
                        date_to_dialogs[date].append({
                            "query": current_user_input,
                            "response": "",
                            "timestamp": timestamp
                        })
                    current_user_input = text
                else:
                    # AI发言，完成当前的对话对
                    date_to_dialogs[date].append({
                        "query": current_user_input,
                        "response": text,
                        "timestamp": timestamp
                    })
                    current_user_input = ""  # 重置
            # 处理session结束后可能的未配对用户发言
            if current_user_input:
                date_to_dialogs[date].append({
                    "query": current_user_input,
                    "response": "",
                    "timestamp": timestamp
                })

        # 可选：统计时间分布，帮助选择校准因子
        all_dates = list(date_to_dialogs.keys())
        if all_dates:
            date_diffs = []
            for date in all_dates:
                try:
                    days = self._get_date_difference(date, now_date)
                    date_diffs.append(days)
                except:
                    pass
            if date_diffs:
                avg_t = sum(date_diffs) / len(date_diffs)
                print(f"样本时间统计: 平均天数差={avg_t:.0f}天")
                # 根据平均天数差推荐校准因子
                recommended_cf = avg_t / (5 * math.log(1 / self.target_retention))
                print(f"推荐校准因子: {recommended_cf:.1f}")

        # 第二步：将整理好的数据放入memory_bank并应用遗忘机制
        for date, dialogs in date_to_dialogs.items():
            memory_str = f'时间{date}的对话内容：' if self.language == 'cn' else f'Conversation content on {date}:'
            user_kw = '[|用户|]：' if self.language == 'cn' else '[|User|]:'
            ai_kw = '[|AI恋人|]：' if self.language == 'cn' else '[|AI|]:'
            # 需要遗忘的记忆索引
            forget_ids = []
            # 初始化该日期的对话列表
            self.memory_bank[sample_id]['history'][date] = []
            # 处理每日的每个对话（仿照原项目逻辑）
            for i, dialog in enumerate(dialogs):
                # 构建记忆数据（仿照原项目格式）
                memory_data = {
                    "query": dialog["query"].strip(),
                    "response": dialog["response"].strip(),
                    "memory_strength": 1.0,  # 初始记忆强度
                    "last_recall_date": date,  # 初始为对话日期
                    "memory_id": f'{sample_id}_{date}_{i}',  # 唯一ID
                    "timestamp": dialog.get("timestamp", "")
                }
                self.memory_bank[sample_id]['history'][date].append(memory_data)
                # 构建记忆文本
                tmp_str = memory_str
                if memory_data["query"]:
                    tmp_str += f'{user_kw} {memory_data["query"]}'
                    if memory_data["response"]:
                        tmp_str += '; '
                if memory_data["response"]:
                    tmp_str += f'{ai_kw} {memory_data["response"]}'

                # 应用校准后的遗忘曲线
                days_diff = self._get_date_difference(date, now_date)
                # 使用校准后的遗忘曲线
                retention_probability = forgetting_curve_calibrated(
                    days_diff,
                    memory_data["memory_strength"],
                    self.calibration_factor
                )

                print(f"日期: {date}, 天数差: {days_diff}, 校准因子: {self.calibration_factor}, "
                      f"保留概率: {retention_probability:.4f}")

                # 随机决定是否遗忘
                if random.random() > retention_probability:
                    forget_ids.append(i)
                    print(f"  记忆 {memory_data['memory_id']} 将被遗忘")
                else:
                    # 保留的记忆，创建Document
                    metadata = {
                        'memory_strength': memory_data["memory_strength"],
                        'memory_id': memory_data["memory_id"],
                        'last_recall_date': memory_data["last_recall_date"],
                        "source": date,  # source设为日期，便于后续检索
                        "date": date,
                        "sample_id": sample_id
                    }
                    docs.append(Document(page_content=tmp_str, metadata=metadata))

            # 处理需要遗忘的记忆（从后往前删除，仿照原项目）
            print(f"样本 {sample_id}, 日期 {date}, 遗忘 {len(forget_ids)}/{len(dialogs)} 条记忆")
            if len(forget_ids) > 0:
                forget_ids.sort(reverse=True)
                for idd in forget_ids:
                    self.memory_bank[sample_id]['history'][date].pop(idd)
            # 如果该日期所有记忆都被遗忘，清理空日期
            if len(self.memory_bank[sample_id]['history'][date]) == 0:
                self.memory_bank[sample_id]['history'].pop(date)

        # 第三步：处理summary（如果有的话）
        # 由于locomo没有summary，这里保持原项目逻辑但跳过
        if 'summary' in self.memory_bank[sample_id].keys():
            # 原项目的summary处理逻辑（但locomo没有）
            pass

        # 第四步：保存记忆（写入文件）- 可选，如果想保存遗忘后的状态
        # 这里可以控制是否保存
        # output_file = self.filepath.replace('.json', f'_{sample_id}_memorybank.json')
        # self.write_memories(output_file)

        print(f"处理完成：总共处理了 {sum(len(d) for d in date_to_dialogs.values())} 个对话")
        print(f"保留记忆：{len(docs)} 个 (保留率: {len(docs) / sum(len(d) for d in date_to_dialogs.values()):.2f})")
        return docs


def get_docs_with_score(docs_with_score):
    # 把(文档，分数)元组列表转换为文档列表，并将分数添加到文档元数据中
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    # 把连续整数列表分割成连续子序列
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def load_memory_file(filepath, user_name, cur_date='', language='cn', calibration_factor=None):
    memory_loader = MemoryForgetterLoader(filepath, language)

    print('加载文件:', filepath)
    if not cur_date:
        cur_date = datetime.date.today().strftime("%Y-%m-%d")
    # 这里的user-name是样本索引
    docs = memory_loader.initial_load_forget_and_save(
        user_name,
        cur_date,
        calibration_factor=calibration_factor
    )
    # docs = textsplitter.split_documents(docs)

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
        for k in range(1, max(i, len(docs) - i)):
            for l in [i + k, i - k]:
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
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn',
                 calibration_factor=100  # 添加校准因子参数 200->100
                 ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.user = ''
        self.top_k = top_k
        self.memory_loader = None
        self.memory_path = ''
        self.calibration_factor = calibration_factor  # 保存校准因子

    def init_memory_vector_store(self,
                                 filepath: str or List[str],
                                 vs_path: str or os.PathLike = None,
                                 user_name: int = None,
                                 cur_date: str = None):
        # 初始化记忆向量存储
        loaded_files = []
        filepath = Path(filepath)
        # filepath = filepath.replace('user',user_name)
        # vs_path = vs_path.replace('user',user_name)
        self.user = user_name
        if not filepath.exists():
            print("路径不存在")
            return None, None
        elif filepath.is_file():
            # 把文件内容读出来，装进document
            docs, self.memory_loader = load_memory_file(
                str(filepath),
                user_name,
                cur_date,
                self.language,
                calibration_factor=self.calibration_factor  # 传递校准因子
            )
            print(f"{filepath.name} 已成功加载")
            # print("locomo版本的docs：")
            # print(docs)
            loaded_files.append(filepath)
        elif filepath.is_dir():
            docs = []
            for child in filepath.iterdir():
                try:
                    new_docs, loader = load_memory_file(
                        str(child),
                        user_name,
                        cur_date,
                        self.language,
                        calibration_factor=self.calibration_factor
                    )
                    docs += new_docs
                    self.memory_loader = loader  # 保存最后一个loader
                    print(f"{child.name} 已成功加载")
                    loaded_files.append(str(child))
                except Exception as e:
                    print(e)
                    print(f"{child.name} 未能成功加载")

        if len(docs) > 0:
            vs_path = Path(vs_path) if vs_path else None
            if vs_path and vs_path.is_dir():
                # 续库，把新读到的文档插进去，like往旧相册添加新照片
                vector_store = FAISS.load_local(str(vs_path), self.embeddings)
                print(f'从之前的记忆索引位置 {vs_path} 加载.')
                vector_store.add_documents(docs)
            else:
                stem = filepath.stem
                if not vs_path:
                    # 时间戳
                    time_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 建文件
                    vs_path = Path(VS_ROOT_PATH) / f"{stem}_FAISS_{time_suffix}"
                    vs_path.mkdir(parents=True, exist_ok=True)
                vector_store = FAISS.from_documents(docs, self.embeddings)
            print(f'保存至 {vs_path}')
            vector_store.save_local(str(vs_path))
            return str(vs_path), loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def load_memory_index(self, vs_path):
        # 加载已保存的记忆向量索引
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        return vector_store

    # def search_memory(self,
    #                   query,
    #                   vector_store,
    #                   cur_date=''):
    #     # 搜索记忆
    #     related_docs_with_score = vector_store.similarity_search_with_score(query,
    #                                                                         k=self.top_k)
    #     related_docs = get_docs_with_score(related_docs_with_score)
    #     related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
    #     pre_date = ''
    #     date_docs = []
    #     dates = []
    #     cur_date = cur_date if cur_date else datetime.date.today().strftime("%Y-%m-%d")
    #     for doc in related_docs:
    #         # saving updated memory
    #         # 从metadata中获取日期
    #         source_date = doc.metadata.get("date", doc.metadata.get("source", ""))
    #         doc.page_content = doc.page_content.replace(f'时间{source_date}的对话内容：', '').strip()
    #         if source_date != pre_date:
    #             date_docs.append(doc.page_content)
    #             pre_date = source_date
    #             dates.append(source_date)
    #         else:
    #             date_docs[-1] += f'\n{doc.page_content}'
    #
    #     # 如果存在memory_loader，更新记忆强度
    #     if hasattr(self, 'memory_loader') and self.memory_loader:
    #         self.memory_loader.update_memory_when_searched(related_docs, user=self.user, cur_date=cur_date)
    #         self.save_updated_memory()
    #
    #     return date_docs, ', '.join(dates)

    def search_memory(self,
                      query,
                      vector_store,
                      cur_date='',
                      max_dates=3,
                      max_chars_per_date=300):
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
            # saving updated memory
            # 从metadata中获取日期
            source_date = doc.metadata.get("date", doc.metadata.get("source", ""))
            doc.page_content = doc.page_content.replace(f'时间{source_date}的对话内容：', '').strip()
            if source_date != pre_date:
                date_docs.append(doc.page_content)
                pre_date = source_date
                dates.append(source_date)
            else:
                date_docs[-1] += f'\n{doc.page_content}'

        # 限制返回的日期数量
        if len(date_docs) > max_dates:
            date_docs = date_docs[:max_dates]
            dates = dates[:max_dates]
        # 限制每个日期的长度
        for i in range(len(date_docs)):
            if len(date_docs[i]) > max_chars_per_date:
                date_docs[i] = date_docs[i][:max_chars_per_date] + "..."
        # 如果存在memory_loader，更新记忆强度
        if hasattr(self, 'memory_loader') and self.memory_loader:
            self.memory_loader.update_memory_when_searched(related_docs, user=self.user, cur_date=cur_date)
            self.save_updated_memory()
        return date_docs, ', '.join(dates)

    def save_updated_memory(self):
        # 保存更新后的记忆到文件（可选）
        if hasattr(self, 'memory_loader') and self.memory_loader:
            # 可以选择是否保存
            # self.memory_loader.write_memories(self.memory_path)
            pass