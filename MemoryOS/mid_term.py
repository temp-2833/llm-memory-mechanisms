import json
import numpy as np
from collections import defaultdict
import faiss
import heapq
from datetime import datetime

try:
    from .utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient,
        llm_extract_keywords
    )
except ImportError:
    from utils import (
        get_timestamp, generate_id, get_embedding, normalize_vector, 
        compute_time_decay, ensure_directory_exists, OpenAIClient,
        llm_extract_keywords
    )

HEAT_ALPHA = 1.0
HEAT_BETA = 1.0
HEAT_GAMMA = 1
RECENCY_TAU_HOURS = 24  # For R_recency calculation in compute_segment_heat


def compute_segment_heat(session, alpha=HEAT_ALPHA, beta=HEAT_BETA, gamma=HEAT_GAMMA, tau_hours=RECENCY_TAU_HOURS):
    # 这个函数是计算一个session的热度，越高保留
    # 公式是α·访问次数 + β·交互页数 + γ·时间衰减因子
    N_visit = session.get("N_visit", 0)  # 访问次数
    L_interaction = session.get("L_interaction", 0)
    
    # Calculate recency based on last_visit_time
    R_recency = 1.0  # Default if no last_visit_time
    if session.get("last_visit_time"):
        # 事件距离当前时间点的远近，越大越近
        R_recency = compute_time_decay(session["last_visit_time"], get_timestamp(), tau_hours)
    
    session["R_recency"] = R_recency  # Update session's recency factor
    # 公式计算并返回
    return alpha * N_visit + beta * L_interaction + gamma * R_recency


class MidTermMemory:
    def __init__(self, file_path: str, client: OpenAIClient, max_capacity=2000, embedding_model_name: str = "all-MiniLM-L6-v2", embedding_model_kwargs: dict = None):
        self.file_path = file_path
        ensure_directory_exists(self.file_path)
        self.client = client
        self.max_capacity = max_capacity  # 最大容纳多少session
        self.sessions = {}  # {session_id: session_object}
        # 被访问次数
        self.access_frequency = defaultdict(int)  # {session_id: access_count_for_lfu}
        # 热度低的在堆顶
        self.heap = []  # Min-heap storing (-H_segment, session_id) for hottest segments

        self.embedding_model_name = embedding_model_name
        self.embedding_model_kwargs = embedding_model_kwargs if embedding_model_kwargs is not None else {}
        self.load()

    def get_page_by_id(self, page_id):
        # 根据page_id在session中寻找页
        for session in self.sessions.values():
            for page in session.get("details", []):
                if page.get("page_id") == page_id:
                    return page
        return None

    def update_page_connections(self, prev_page_id, next_page_id):
        # 把给定的两个页做成前后页，便于上一页下一页跳转
        if prev_page_id:
            # 调用上面的函数
            prev_page = self.get_page_by_id(prev_page_id)
            if prev_page:
                prev_page["next_page"] = next_page_id
        if next_page_id:
            next_page = self.get_page_by_id(next_page_id)
            if next_page:
                next_page["pre_page"] = prev_page_id
        # self.save() # Avoid saving on every minor update; save at higher level operations

    def evict_lfu(self):
        # LFU淘汰策略(按照访问次数多少淘汰)
        if not self.access_frequency or not self.sessions:
            return

        # 找到访问次数最少的session_id
        lfu_sid = min(self.access_frequency, key=self.access_frequency.get)
        print(f"MTM: LFU 策略淘汰. Session {lfu_sid} 的访问次数最低.")
        
        if lfu_sid not in self.sessions:  # 但此id已经不在session中
            del self.access_frequency[lfu_sid]  # Clean up access frequency if session already gone
            self.rebuild_heap()
            return
        # 获得要淘汰的session
        session_to_delete = self.sessions.pop(lfu_sid)  # Remove from sessions
        # 从记录所有session访问次数的字典中去除
        del self.access_frequency[lfu_sid]  # Remove from LFU tracking

        for page in session_to_delete.get("details", []):
            prev_page_id = page.get("pre_page")
            next_page_id = page.get("next_page")
            # If a page from this session was linked to an external page, nullify the external link
            if prev_page_id and not self.get_page_by_id(prev_page_id):
                # Check if prev page is still in memory
                 pass 
            if next_page_id and not self.get_page_by_id(next_page_id):
                 pass
            # More robustly, one might need to search all other sessions if inter-session linking was allowed
            # For now, assuming internal consistency or that MemoryOS class manages higher-level links

        self.rebuild_heap()
        self.save()
        print(f"MTM: 淘汰 session {lfu_sid}.")

    def add_session(self, summary, details, summary_keywords=None):
        # 添加session及其中的pages
        session_id = generate_id("session")
        # 摘要向量
        summary_vec = get_embedding(
            summary, 
            model_name=self.embedding_model_name, 
            **self.embedding_model_kwargs
        )
        # 单位化
        summary_vec = normalize_vector(summary_vec).tolist()
        # 摘要关键字(可为空)
        summary_keywords = summary_keywords if summary_keywords is not None else []
        
        processed_details = []  # 记录每一页的信息、id、关键词等
        for page_data in details:  # details是此session下的所有page
            # 每一页的id get函数的意思是，如果page_id存在则使用，不存在则generate
            page_id = page_data.get("page_id", generate_id("page"))
            
            # 检查是否已有embedding，避免重复计算
            if "page_embedding" in page_data and page_data["page_embedding"]:
                print(f"MTM: 再次使用已经存在的embedding给page {page_id}")
                inp_vec = page_data["page_embedding"]
                # 确保embedding是normalized的
                if isinstance(inp_vec, list):  # 如果inp_vec是列表的话
                    inp_vec_np = np.array(inp_vec, dtype=np.float32)
                    if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                        inp_vec = normalize_vector(inp_vec_np).tolist()
            else:
                # 没有embedding则计算
                # print(f"MTM: 计算新embedding给page {page_id}")
                full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                inp_vec = get_embedding(
                    full_text,
                    model_name=self.embedding_model_name,
                    **self.embedding_model_kwargs
                )
                inp_vec = normalize_vector(inp_vec).tolist()
            
            # 使用已有keywords或设置为空（由multi-summary提供）
            if "page_keywords" in page_data and page_data["page_keywords"]:
                print(f"MTM: 使用已经存在的关键词给page {page_id}")
                page_keywords = page_data["page_keywords"]
            else:
                # print(f"MTM: 设置空的关键词给page {page_id} (会被multi-summary填充)")
                # 修改这里，page也会有keywords
                # page_keywords = []
                page_keywords = summary_keywords

            # 整合当前正在处理的page的所有信息
            processed_page = {
                **page_data,  # 复制已有(QA)字段：user_input, agent_response, timestamp
                "page_id": page_id,
                "page_embedding": inp_vec,
                "page_keywords": page_keywords,
                "preloaded": page_data.get("preloaded", False),  # Preserve if passed
                "analyzed": page_data.get("analyzed", False),   # Preserve if passed
                # pre_page, next_page, meta_info are handled by DynamicUpdater
            }
            processed_details.append(processed_page)
        
        current_ts = get_timestamp()
        session_obj = {  # 当前要添加的session的全部信息
            "id": session_id,
            "summary": summary,
            "summary_keywords": summary_keywords,
            "summary_embedding": summary_vec,
            "details": processed_details,
            "L_interaction": len(processed_details),
            "R_recency": 1.0,  # 初始化recency
            "N_visit": 0,
            "H_segment": 0.0,  # 初始化heat, will be computed
            "timestamp": current_ts,  # Creation timestamp
            "last_visit_time": current_ts,  # Also initial last_visit_time for recency calc
            "access_count_lfu": 0  # For LFU eviction policy
        }
        # 计算当前session的热度
        session_obj["H_segment"] = compute_segment_heat(session_obj)
        self.sessions[session_id] = session_obj
        # 这个是session！的访问次数字典
        self.access_frequency[session_id] = 0  # Initialize for LFU
        # 把session热度放到堆里
        heapq.heappush(self.heap, (-session_obj["H_segment"], session_id))  # Use negative heat for max-heap behavior

        print(f"MTM: 添加新的 session {session_id}. 初始热度值: {session_obj['H_segment']:.2f}.")
        if len(self.sessions) > self.max_capacity:
            self.evict_lfu()  # 如果session队列满了则淘汰
        self.save()
        return session_id  # 返回新增的session_id

    def rebuild_heap(self):
        # 重建最小堆(低热度在堆顶)
        self.heap = []  # 清空
        for sid, session_data in self.sessions.items():
            # 遍历所有session，按照热度插入堆中
            # session_data["H_segment"] = compute_segment_heat(session_data)
            heapq.heappush(self.heap, (-session_data["H_segment"], sid))
        # heapq.heapify(self.heap) # Not needed if pushing one by one
        # No save here, it's an internal operation often followed by other ops that save

    def insert_pages_into_session(self, summary_for_new_pages, keywords_for_new_pages, pages_to_insert, 
                                  similarity_threshold=0.6, keyword_similarity_alpha=1.0):
        # 把页面pages插入到合适的session中
        # 参数：新页面的摘要、关键词、数据列表、相似度阈值、关键词相似度系数
        if not self.sessions:  # If no existing sessions, just add as a new one
            print("MTM：没有存在的sessions，添加新的session")
            # session为空，则创建一个session
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

        # 先计算这些pages的摘要的embedding
        new_summary_vec = get_embedding(
            summary_for_new_pages,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        new_summary_vec = normalize_vector(new_summary_vec)

        # 在已有的session中寻找最佳session
        best_sid = None  # 最佳id
        best_overall_score = -1  # 最佳得分
        # 遍历所有session的id
        for sid, existing_session in self.sessions.items():
            existing_summary_vec = np.array(existing_session["summary_embedding"], dtype=np.float32)
            # 计算当前session的摘要和要加进来的页面的摘要的余弦相似度
            semantic_sim = float(np.dot(existing_summary_vec, new_summary_vec))

            # 基于Jaccard指数计算关键词相似度
            # Keyword similarity (Jaccard index based)
            existing_keywords = set(existing_session.get("summary_keywords", []))
            new_keywords_set = set(keywords_for_new_pages)  # 这是新page的关键词
            s_topic_keywords = 0  # 相似度初始化
            if existing_keywords and new_keywords_set:
                # 交集大小
                intersection = len(existing_keywords.intersection(new_keywords_set))
                # 并集大小
                union = len(existing_keywords.union(new_keywords_set))
                if union > 0:
                    # 相似度
                    s_topic_keywords = intersection / union 

            # 当前session的综合得分
            overall_score = semantic_sim + keyword_similarity_alpha * s_topic_keywords
            # 找到得分最高的那个session
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_sid = sid

        # 存在最佳session且相似度分数大于阈值的话，就正式开始插入
        if best_sid and best_overall_score >= similarity_threshold:
            print(f"MTM: 把pages合并到session {best_sid}. 分数为: {best_overall_score:.2f} (阈值为: {similarity_threshold})")
            target_session = self.sessions[best_sid]  # 目标session
            
            processed_new_pages = []
            # 下面这一行记录已经存在的page的ids，避免重复
            existing_page_ids = {p.get("page_id") for p in target_session.get("details", [])}
            for page_data in pages_to_insert:
                if page_data.get("page_id") in existing_page_ids:
                    print("!!!跳过此重复page")
                    continue

                page_id = page_data.get("page_id", generate_id("page"))
                
                # 检查是否已有embedding，避免重复计算
                if "page_embedding" in page_data and page_data["page_embedding"]:
                    print(f"MTM: 使用已有的embedding给page {page_id}")
                    inp_vec = page_data["page_embedding"]
                    # 确保embedding是normalized的
                    if isinstance(inp_vec, list):
                        inp_vec_np = np.array(inp_vec, dtype=np.float32)
                        if np.linalg.norm(inp_vec_np) > 1.1 or np.linalg.norm(inp_vec_np) < 0.9:  # 检查是否需要重新normalize
                            inp_vec = normalize_vector(inp_vec_np).tolist()
                else:
                    # print(f"MTM: 计算新的embedding给page {page_id}")
                    full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
                    inp_vec = get_embedding(
                        full_text,
                        model_name=self.embedding_model_name,
                        **self.embedding_model_kwargs
                    )
                    inp_vec = normalize_vector(inp_vec).tolist()
                
                # 使用已有keywords或继承session的keywords
                if "page_keywords" in page_data and page_data["page_keywords"]:
                    print(f"MTM: 使用已经存在的关键词给 page {page_id}")
                    page_keywords_current = page_data["page_keywords"]
                else:
                    # print(f"MTM: 使用session的关键词给 page {page_id}")
                    page_keywords_current = keywords_for_new_pages

                # 每一页的具体信息
                processed_page = {
                    **page_data,  # 复制原本有的字段
                    "page_id": page_id,
                    "page_embedding": inp_vec,
                    "page_keywords": page_keywords_current,
                    # analyzed, preloaded flags should be part of page_data if set
                }
                # 把每一页的信息加入目标session
                target_session["details"].append(processed_page)
                processed_new_pages.append(processed_page)

            # 下面更新一些目标session的数据
            target_session["L_interaction"] += len(pages_to_insert)
            target_session["last_visit_time"] = get_timestamp()
            target_session["H_segment"] = compute_segment_heat(target_session)
            self.rebuild_heap()  # Rebuild heap as heat has changed
            self.save()
            return best_sid
        else:
            print(f"MTM: 没有合适的session合并 (最佳分数是 {best_overall_score:.2f} < 阈值 {similarity_threshold}). 创建新session.")
            return self.add_session(summary_for_new_pages, pages_to_insert, keywords_for_new_pages)

    def search_sessions(self, query_text, query_keywords, segment_similarity_threshold=0.1, page_similarity_threshold=0.1,
                          top_k_sessions=5, keyword_alpha=1.0, recency_tau_search=3600):
        # 在session们中搜索指定session，FAISS搜索是Facebook的一个库
        # 参数：查询文本，session相似度阈值，页面相似度，top k个session，关键词参数，时间衰减常数
        if not self.sessions:
            return []

        # 对查询文本做嵌入
        query_vec = get_embedding(
            query_text,
            model_name=self.embedding_model_name,
            **self.embedding_model_kwargs
        )
        query_vec = normalize_vector(query_vec)
        # 下面的英文意思是查询关键词已经剔除，目前仅依赖语义相似度
        # query_keywords = set()  # Keywords extraction removed, relying on semantic similarity
        query_keywords = query_keywords
        print(f"llm生成的关键词内容：{query_keywords}")

        # 候选session
        candidate_sessions = []
        session_ids = list(self.sessions.keys())  # 变成列表
        if not session_ids: return []

        # 所有session的摘要嵌入，放到列表里
        summary_embeddings_list = [self.sessions[s]["summary_embedding"] for s in session_ids]
        summary_embeddings_np = np.array(summary_embeddings_list, dtype=np.float32)

        dim = summary_embeddings_np.shape[1]
        # 使用内积计算相似度，这里的这种维度、faiss使用之前有用过(long_term那里)
        index = faiss.IndexFlatIP(dim)  # Inner product for similarity
        index.add(summary_embeddings_np)
        
        query_arr_np = np.array([query_vec], dtype=np.float32)

        distances, indices = index.search(query_arr_np, min(top_k_sessions, len(session_ids)))

        results = []
        current_time_str = get_timestamp()

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            session_id = session_ids[idx]
            session = self.sessions[session_id]
            # 获取每个session的得分
            semantic_sim_score = float(distances[0][i])  # This is the dot product

            session_keywords = set(session.get("summary_keywords", []))
            s_topic_keywords = 0
            if query_keywords and session_keywords:
                intersection = len(query_keywords.intersection(session_keywords))
                union = len(query_keywords.union(session_keywords))
                if union > 0: s_topic_keywords = intersection / union

            # Time decay for session recency in search scoring
            # time_decay_factor = compute_time_decay(session["timestamp"], current_time_str, tau_hours=recency_tau_search)

            # Combined score for session relevance
            session_relevance_score = (semantic_sim_score + keyword_alpha * s_topic_keywords)
            if session_relevance_score >= segment_similarity_threshold:
                matched_pages_in_session = []
                for page in session.get("details", []):
                    page_embedding = np.array(page["page_embedding"], dtype=np.float32)
                    # page_keywords = set(page.get("page_keywords", []))
                    # 计算当前页面的嵌入和query的相似度
                    page_sim_score = float(np.dot(page_embedding, query_vec))
                    # Can also add keyword sim for pages if needed, but keeping it simpler for now
                    # 页面相似度高于阈值则加入列表
                    if page_sim_score >= page_similarity_threshold:
                        matched_pages_in_session.append({"page_data": page, "score": page_sim_score})
                
                if matched_pages_in_session:  # 存在匹配要求的页面
                    # Update session access stats
                    session["N_visit"] += 1  # 访问次数
                    session["last_visit_time"] = current_time_str
                    # LFU计数+1
                    session["access_count_lfu"] = session.get("access_count_lfu", 0) + 1
                    self.access_frequency[session_id] = session["access_count_lfu"]
                    session["H_segment"] = compute_segment_heat(session)
                    self.rebuild_heap() # Heat changed
                    # 把满足要求的页面按分数排序放进最终结果
                    results.append({
                        "session_id": session_id,
                        "session_summary": session["summary"],
                        "session_relevance_score": session_relevance_score,
                        # Sort pages by score
                        "matched_pages": sorted(matched_pages_in_session, key=lambda x: x["score"], reverse=True)
                    })
        
        self.save()  # Save changes from access updates
        # Sort final results by session_relevance_score
        # 按照session相似度返回结果
        return sorted(results, key=lambda x: x["session_relevance_score"], reverse=True)

    def save(self):
        # 保存到JSON文件
        # Make a copy for saving to avoid modifying heap during iteration if it happens
        # Though current heap is list of tuples, so direct modification risk is low
        # sessions_to_save = {sid: data for sid, data in self.sessions.items()}
        data_to_save = {  # 保存session及其访问次数
            "sessions": self.sessions,
            "access_frequency": dict(self.access_frequency), # Convert defaultdict to dict for JSON
            # Heap is derived, no need to save typically, but can if desired for faster load
            # "heap_snapshot": self.heap 
        }
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Error saving MidTermMemory to {self.file_path}: {e}")

    def load(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.sessions = data.get("sessions", {})
                self.access_frequency = defaultdict(int, data.get("access_frequency", {}))
                self.rebuild_heap()  # Rebuild heap from loaded sessions
            print(f"MTM: 加载记忆from文件 {self.file_path}. Sessions: {len(self.sessions)}.")
        except FileNotFoundError:
            print(f"MidTermMemory: No history file found at {self.file_path}. Initializing new memory.")
        except json.JSONDecodeError:
            print(f"MidTermMemory: Error decoding JSON from {self.file_path}. Initializing new memory.")
        except Exception as e:
            print(f"MidTermMemory: An unexpected error occurred during load from {self.file_path}: {e}. Initializing new memory.") 