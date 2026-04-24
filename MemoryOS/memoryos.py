import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:

    from .utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, \
        ensure_directory_exists, llm_extract_keywords
    from . import prompts
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory, compute_segment_heat  # For H_THRESHOLD logic
    from .long_term import LongTermMemory
    from .updater import Updater
    from .retriever import Retriever
except ImportError:

    from utils import OpenAIClient, get_timestamp, generate_id, gpt_user_profile_analysis, gpt_knowledge_extraction, \
        ensure_directory_exists, llm_extract_keywords
    import prompts
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory, compute_segment_heat  # For H_THRESHOLD logic
    from long_term import LongTermMemory
    from updater import Updater
    from retriever import Retriever

# 触发MTM到LTM的画像/知识更新的热度阈值
H_PROFILE_UPDATE_THRESHOLD = 5.0
DEFAULT_ASSISTANT_ID = "default_assistant_profile"


class Memoryos:
    # memory os
    def __init__(self, user_id: str,
                 openai_api_key: str,
                 data_storage_path: str,
                 openai_base_url: str = None,
                 assistant_id: str = DEFAULT_ASSISTANT_ID,
                 short_term_capacity=10,
                 mid_term_capacity=2000,
                 long_term_knowledge_capacity=100,
                 retrieval_queue_capacity=7,
                 mid_term_heat_threshold=H_PROFILE_UPDATE_THRESHOLD,
                 mid_term_similarity_threshold=0.6,
                 llm_model="gpt-4o-mini",

                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 embedding_model_kwargs: dict = None
                 ):
        self.user_id = user_id
        self.assistant_id = assistant_id
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.llm_model = llm_model
        self.mid_term_similarity_threshold = mid_term_similarity_threshold
        self.embedding_model_name = embedding_model_name
        # -------------------------------
        self.cache_qa = []  # 缓存待处理的QA队列
        # ------------------------------

        if embedding_model_kwargs is None:
            if 'bge-m3' in self.embedding_model_name.lower():
                print("信息: 检测到 bge-m3 模型, 默认参数是 embedding_model_kwargs to {'use_fp16': True}")
                self.embedding_model_kwargs = {'use_fp16': True}
            else:
                # 当前走这条路
                self.embedding_model_kwargs = {}
        else:
            self.embedding_model_kwargs = embedding_model_kwargs

        print(
            f"初始化 Memoryos user_id是 '{self.user_id}' 以及助手id： '{self.assistant_id}'. 数据存储路径: {self.data_storage_path}")
        print(f"使用统一llm模型: {self.llm_model}")
        print(f"使用的embedding模型是: {self.embedding_model_name} 以及关键字参数: {self.embedding_model_kwargs}")

        # 初始化AI模型
        self.client = OpenAIClient(api_key=openai_api_key, base_url=openai_base_url)

        # user-specific data 建目录
        self.user_data_dir = os.path.join(self.data_storage_path, "users", self.user_id)
        user_short_term_path = os.path.join(self.user_data_dir, "short_term.json")
        user_mid_term_path = os.path.join(self.user_data_dir, "mid_term.json")
        user_long_term_path = os.path.join(self.user_data_dir,
                                           "long_term_user.json")  # User profile and their knowledge

        # assistant-specific data (knowledge)
        self.assistant_data_dir = os.path.join(self.data_storage_path, "assistants", self.assistant_id)
        assistant_long_term_path = os.path.join(self.assistant_data_dir, "long_term_assistant.json")

        # 确保目录存在
        ensure_directory_exists(user_short_term_path)
        ensure_directory_exists(user_mid_term_path)
        ensure_directory_exists(user_long_term_path)
        ensure_directory_exists(assistant_long_term_path)

        # 给STM、MTM、LTM三个类实例化一下
        self.short_term_memory = ShortTermMemory(file_path=user_short_term_path, max_capacity=short_term_capacity)
        self.mid_term_memory = MidTermMemory(
            file_path=user_mid_term_path,
            client=self.client,
            max_capacity=mid_term_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )
        self.user_long_term_memory = LongTermMemory(
            file_path=user_long_term_path,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        self.assistant_long_term_memory = LongTermMemory(
            file_path=assistant_long_term_path,
            knowledge_capacity=long_term_knowledge_capacity,
            embedding_model_name=self.embedding_model_name,
            embedding_model_kwargs=self.embedding_model_kwargs
        )

        # 初始化更新和检索模块
        self.updater = Updater(short_term_memory=self.short_term_memory,
                               mid_term_memory=self.mid_term_memory,
                               long_term_memory=self.user_long_term_memory,
                               # Updater primarily updates user's LTM profile/knowledge
                               client=self.client,
                               topic_similarity_threshold=mid_term_similarity_threshold,  # 传递中期记忆相似度阈值
                               llm_model=self.llm_model)
        self.retriever = Retriever(
            mid_term_memory=self.mid_term_memory,
            long_term_memory=self.user_long_term_memory,
            assistant_long_term_memory=self.assistant_long_term_memory,  # Pass assistant LTM
            queue_capacity=retrieval_queue_capacity
        )
        # MTM的热度阈值
        self.mid_term_heat_threshold = mid_term_heat_threshold

    def _trigger_profile_and_knowledge_update_if_needed(self, speaker_a="", speaker_b=""):
        # 检查中期记忆中最热的session，如果超过阈值则更新画像和知识
        if not self.mid_term_memory.heap:
            return
        # MidTermMemory heap stores (-H_segment, sid)
        neg_heat, sid = self.mid_term_memory.heap[0]
        current_heat = -neg_heat  # 当前最高的热度
        # 每次只更新一个session？
        if current_heat >= self.mid_term_heat_threshold:
            # 跟阈值比较
            # 大的话则获取这个heat最大的session
            session = self.mid_term_memory.sessions.get(sid)
            if not session:
                self.mid_term_memory.rebuild_heap()  # Clean up if session is gone
                return
            # A page is a dict: {"user_input": ..., "agent_response": ..., "timestamp": ..., "analyzed": False, ...}
            unanalyzed_pages = [p for p in session.get("details", []) if not p.get("analyzed", False)]
            if unanalyzed_pages:
                print(
                    f"Memoryos: MTM的session {sid} 热度 ({current_heat:.2f}) 超出阈值. 为了画像/知识更新分析 {len(unanalyzed_pages)} 张pages.")
                # 并行执行两个LLM任务：用户画像分析（已包含更新）、知识提取

                def task_user_profile_analysis():
                    print("Memoryos: 开始并行提取用户画像...")
                    # 获取现有用户画像
                    existing_profile = self.user_long_term_memory.get_raw_user_profile(self.user_id)
                    if not existing_profile or existing_profile.lower() == "none":
                        existing_profile = "No existing profile data."
                    # 直接输出更新后的完整画像
                    return gpt_user_profile_analysis(unanalyzed_pages, self.client, model=self.llm_model,
                                                     existing_user_profile=existing_profile)

                def task_knowledge_extraction():
                    print("Memoryos: 开始并行知识提取...")
                    return gpt_knowledge_extraction(unanalyzed_pages, self.client, model=self.llm_model)

                # 使用并行任务执行
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # 提交两个主要任务
                    future_profile = executor.submit(task_user_profile_analysis)
                    future_knowledge = executor.submit(task_knowledge_extraction)
                    # 等待结果
                    try:
                        updated_user_profile = future_profile.result()  # 直接是更新后的完整画像
                        knowledge_result = future_knowledge.result()
                    except Exception as e:
                        print(f"Error in parallel LLM processing: {e}")
                        return

                new_user_private_knowledge = knowledge_result.get("private")
                new_assistant_knowledge = knowledge_result.get("assistant_knowledge")

                # 直接使用更新后的完整用户画像
                if updated_user_profile and updated_user_profile.lower() != "none":
                    print("Memoryos: Updating user profile with integrated analysis...")
                    self.user_long_term_memory.update_user_profile(self.user_id, updated_user_profile,
                                                                   merge=False)  # 直接替换为新的完整画像

                # 把用户隐私添加到关于用户的长期记忆中
                if new_user_private_knowledge and new_user_private_knowledge.lower() != "none":
                    for line in new_user_private_knowledge.split('\n'):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.user_long_term_memory.add_user_knowledge(line.strip())
                # 把助手的知识添加到它的长期记忆中
                if new_assistant_knowledge and new_assistant_knowledge.lower() != "none":
                    for line in new_assistant_knowledge.split('\n'):
                        if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                            self.assistant_long_term_memory.add_assistant_knowledge(
                                line.strip())  # Save to dedicated assistant LTM

                for p in session["details"]:
                    p["analyzed"] = True
                # 下面重置访问次数、时效性、热度等
                session["N_visit"] = 0  # 访问次数
                session["L_interaction"] = 0  # 包含页数
                # session["R_recency"] = 1.0 # Recency will re-calculate naturally
                session["H_segment"] = compute_segment_heat(session)  # 热度
                session["last_visit_time"] = get_timestamp()

                self.mid_term_memory.rebuild_heap()
                self.mid_term_memory.save()
                print(f"Memoryos: 已完成session {sid}的画像/知识更新. 热度已重置.")
            else:
                print(f"Memoryos: 高热度的session {sid} 没有未分析的pages. 跳过画像更新.")
        else:
            pass

    def add_memory(self, user_input: str, agent_response: str, timestamp: str = None, speaker_a="", speaker_b="",
                   meta_data: dict = None):
        """
        添加新的QA对到STM，满了则搬到MTM
        meta_data is not used in the current refactoring but kept for future use.
        """
        if not timestamp:
            timestamp = get_timestamp()

        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp
            # meta_data can be added here if it needs to be stored with the QA pair
        }
        self.short_term_memory.add_qa_pair(qa_pair)
        print(f"Memoryos: 给STM添加QA. User输入为: {user_input[:30]}...")

        if self.short_term_memory.is_full():
            print("Memoryos: STM已满. 需要转移到MTM.")

            self.updater.process_short_term_to_mid_term()
            # ------------------------------------
            # qa = self.short_term_memory.pop_oldest()

        self._trigger_profile_and_knowledge_update_if_needed(speaker_a=speaker_a, speaker_b=speaker_b)

    def get_response(self, query: str, relationship_with_user="friend", speaker_a="", speaker_b="", style_hint="",
                     user_conversation_meta_data: dict = None) -> str:
        # 生成给user的回答
        # print(f"Memoryos: 生成针对: '{query[:50]}...' 的回答")
        print(f"Memoryos: 生成针对: '{query}' 的回答")
        # ------------------------ 补充query关键词
        query_keywords = llm_extract_keywords(query, self.client, self.llm_model)

        retrieval_results = self.retriever.retrieve_context(
            user_query=query,
            query_keywords=query_keywords,
            user_id=self.user_id
            # 暂时使用Retriever类中的默认阈值
        )
        retrieved_pages = retrieval_results["retrieved_pages"]
        retrieved_user_knowledge = retrieval_results["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval_results["retrieved_assistant_knowledge"]
        # 2. 获取全部短期记忆历史
        short_term_history = self.short_term_memory.get_all()
        history_text = "\n".join([
            f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')} (Time: {qa.get('timestamp', '')})"
            for qa in short_term_history
        ])
        # print(f"获取到的STM内容为：{history_text}")
        # 3. 规范化检索到的MTM pages(retrieval_queue equivalent?)
        # 中期pages
        retrieval_text = "\n".join([
            f"【Historical Memory】\n{speaker_a}: {page.get('user_input', '')}\n{speaker_b}: {page.get('agent_response', '')}\nTime: {page.get('timestamp', '')}\nConversation chain overview: {page.get('meta_info', 'N/A')}"
            for page in retrieved_pages
        ])
        # print(f"获取到的MTM内容为：{retrieval_text}")
        # 4. 获取用户画像
        user_profile_text = self.user_long_term_memory.get_raw_user_profile(self.user_id)
        if not user_profile_text or user_profile_text.lower() == "none":
            user_profile_text = "No detailed profile available yet."
        # 5. 规范化检索到的用户知识，作为背景信息
        user_knowledge_background = ""
        if retrieved_user_knowledge:
            user_knowledge_background = f"\n【Knowledge about {speaker_a}】\n"
            for kn_entry in retrieved_user_knowledge:
                user_knowledge_background += f"- {kn_entry['knowledge']} (Recorded: {kn_entry['timestamp']})\n"
        background_context = f"【{speaker_a}'s Profile】\n{user_profile_text}\n{user_knowledge_background}"
        # print(f"获取到的user-knowledge和user画像内容为：{background_context}")
        # 6. 规范化检索到的助手知识
        assistant_knowledge_text_for_prompt = f"【Knowledge about {speaker_b}】\n"
        if retrieved_assistant_knowledge:
            for ak_entry in retrieved_assistant_knowledge:
                assistant_knowledge_text_for_prompt += f"- {ak_entry['knowledge']} (Recorded: {ak_entry['timestamp']})\n"
        else:
            assistant_knowledge_text_for_prompt += f"- No relevant knowledge about {speaker_b} found for this query.\n"
        # print(f"获取到的ai-knowledge内容为：{assistant_knowledge_text_for_prompt}")
        # 7. 规范化用户对话的meta_data(如果有的话)
        meta_data_text_for_prompt = "【Current Conversation Metadata】\n"
        if user_conversation_meta_data:
            try:
                meta_data_text_for_prompt += json.dumps(user_conversation_meta_data, ensure_ascii=False, indent=2)
            except TypeError:
                meta_data_text_for_prompt += str(user_conversation_meta_data)
        else:
            meta_data_text_for_prompt += "None provided for this turn."
        # 8. 组装prompts
        # ------------------------locomo-------------------
        system_prompt_text = prompts.system_prompt_locomo.format(
            speaker_b=speaker_b,
            speaker_a=speaker_a,
            assistant_knowledge_text=assistant_knowledge_text_for_prompt
        )
        user_prompt_text = prompts.user_prompt_locomo.format(
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            history_text=history_text,  # STM
            retrieval_text=retrieval_text,  # MTM
            background=background_context,
            query=query
        )
        # -------------------------------------
        # ---------------------GVD-----------------------
        # system_prompt_text = prompts.system_prompt_GVD.format(
        #     # 这边可以提供很多history，但是prompt只需要一个
        #     USER_ID=speaker_a,
        #     assistant_knowledge_text=assistant_knowledge_text_for_prompt
        # )
        # user_prompt_text = prompts.user_prompt_GVD.format(
        #     USER_ID=speaker_a,
        #     history_text=history_text,  # STM
        #     retrieval_text=retrieval_text,  # MTM
        #     background=user_knowledge_background,  # 用户画像和knowledge (修改为只发知识，而没有画像)
        #     query=query
        # )
        # --------------------------------------
        messages = [
            {"role": "system", "content": system_prompt_text},
            {"role": "user", "content": user_prompt_text}
        ]
        # 9. 调用大模型来回答问题
        print("Memoryos: 调用指定llm生成最终的回答...")
        # print("System Prompt:\n", system_prompt_text)
        # print("User Prompt:\n", user_prompt_text)
        estimated_tokens = len(json.dumps(messages)) * 1.3  # 粗略估算
        print(f"📝 get_response请求: 估算{estimated_tokens:.0f}tokens，限制2000")
        # === 在函数末尾添加：将以上所有内容写入文件 ===
        with open("memoryos_debug-conv-50-debug.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"查询: {query}\n")
            f.write("=" * 80 + "\n\n")
            f.write("【短期记忆 STM】\n")
            f.write(history_text + "\n\n")
            f.write("【中期记忆 MTM】\n")
            f.write(retrieval_text + "\n\n")
            f.write("【用户知识】\n")
            f.write(background_context + "\n\n")
            f.write("【助手知识】\n")
            f.write(assistant_knowledge_text_for_prompt + "\n\n")
            f.write("=" * 80 + "\n\n")
        # ------------------------- 检测阶段，先把实际获取回答注释掉
        response_content = self.client.chat_completion(
            model=self.llm_model,
            messages=messages,
            temperature=0.7,  # 0.7->0.0
            max_tokens=2000  # As in original main  1500->2000
        )

        print(f"Memoryos: 问题:'{query[:50]}...'的回答是: {response_content}")
        # --------------------------------
        # 10. 把此次交互记录到记忆中
        # self.add_memory(user_input=query, agent_response=response_content, timestamp=get_timestamp())
        return response_content


    # --- Helper/Maintenance methods (可选的附加项) ---
    def get_user_profile_summary(self) -> str:
        # 获取用户画像
        return self.user_long_term_memory.get_raw_user_profile(self.user_id)

    def get_assistant_knowledge_summary(self) -> list:
        # 获取助手知识库
        return self.assistant_long_term_memory.get_assistant_knowledge()

    def force_mid_term_analysis(self):
        """
        Forces analysis of all unanalyzed pages in the hottest mid-term segment if heat is above 0.
        Useful for testing or manual triggering.
        """
        original_threshold = self.mid_term_heat_threshold
        self.mid_term_heat_threshold = 0.0  # Temporarily lower threshold
        print("Memoryos: 强制触发MTM分析...")
        self._trigger_profile_and_knowledge_update_if_needed()
        self.mid_term_heat_threshold = original_threshold  # Restore original threshold

    def __repr__(self):
        return f"<Memoryos user_id='{self.user_id}' assistant_id='{self.assistant_id}' data_path='{self.data_storage_path}'>"