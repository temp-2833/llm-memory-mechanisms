try:
    from .utils import (
        generate_id, get_timestamp, 
        gpt_generate_multi_summary, check_conversation_continuity, generate_page_meta_info, OpenAIClient
    )
    from .short_term import ShortTermMemory
    from .mid_term import MidTermMemory
    from .long_term import LongTermMemory
except ImportError:
    from utils import (
        generate_id, get_timestamp, 
        gpt_generate_multi_summary, check_conversation_continuity, generate_page_meta_info, OpenAIClient
    )
    from short_term import ShortTermMemory
    from mid_term import MidTermMemory
    from long_term import LongTermMemory

from concurrent.futures import ThreadPoolExecutor, as_completed


class Updater:
    def __init__(self, 
                 short_term_memory: ShortTermMemory, 
                 mid_term_memory: MidTermMemory, 
                 long_term_memory: LongTermMemory, 
                 client: OpenAIClient,
                 topic_similarity_threshold=0.5,  # mid_term
                 llm_model="gpt-4o-mini"):
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.client = client
        self.topic_similarity_threshold = topic_similarity_threshold
        self.last_evicted_page_for_continuity = None  # Tracks the actual last page object for continuity checks
        self.llm_model = llm_model

    def _process_page_embedding_and_keywords(self, page_data):
        page_id = page_data.get("page_id", generate_id("page"))
        # 检查是否已有embedding
        if "page_embedding" in page_data and page_data["page_embedding"]:
            print(f"更新: Page {page_id} 已经有embedding, 跳过计算")
            return page_data

        if not ("page_embedding" in page_data and page_data["page_embedding"]):
            full_text = f"User: {page_data.get('user_input','')} Assistant: {page_data.get('agent_response','')}"
            try:
                embedding = self._get_embedding_for_page(full_text)
                if embedding is not None:
                    from .utils import normalize_vector
                    page_data["page_embedding"] = normalize_vector(embedding).tolist()
                    print(f"更新: 生成embedding给 page {page_id}")
            except Exception as e:
                print(f"Error generating embedding for page {page_id}: {e}")

        if "page_keywords" not in page_data:
            page_data["page_keywords"] = []
        
        return page_data

    def _get_embedding_for_page(self, text):
        # 获取页面embedding的辅助方法
        from .utils import get_embedding
        return get_embedding(text)

    def _update_linked_pages_meta_info(self, start_page_id, new_meta_info):
        # 更新从start_page开始的一连串page的meta信息
        q = [start_page_id]  # 队列
        visited = {start_page_id}  # 已经处理的id
        
        head = 0  # 索引
        while head < len(q):
            current_page_id = q[head]
            head += 1
            # 根据id获取当前page
            page = self.mid_term_memory.get_page_by_id(current_page_id)
            if page:  # 给此page添加新的meta信息
                page["meta_info"] = new_meta_info
                prev_id = page.get("pre_page")
                if prev_id and prev_id not in visited:
                    q.append(prev_id)
                    visited.add(prev_id)
                next_id = page.get("next_page")
                if next_id and next_id not in visited:
                    q.append(next_id)
                    visited.add(next_id)
        if q:
            self.mid_term_memory.save()

    def process_short_term_to_mid_term(self):
        # 把短期记忆转移到中期
        evicted_qas = []
        while self.short_term_memory.is_full():
            qa = self.short_term_memory.pop_oldest()
            if qa and qa.get("user_input") and qa.get("agent_response"):
                evicted_qas.append(qa)
        
        if not evicted_qas:
            print("更新: 没有QAs被STM淘汰.")
            return

        print(f"更新: 处理 {len(evicted_qas)} 个QAs 从STM到MTM.")

        current_batch_pages = []  # 存储每一个被驱逐的qa形成的page
        temp_last_page_in_batch = self.last_evicted_page_for_continuity  # Carry over from previous batch if any

        for qa_pair in evicted_qas:
            # 遍历被驱逐的短期记忆信息
            current_page_obj = {  # 把STM的QA变成page的格式
                "page_id": generate_id("page"),
                "user_input": qa_pair.get("user_input", ""),
                "agent_response": qa_pair.get("agent_response", ""),
                "timestamp": qa_pair.get("timestamp", get_timestamp()),
                "preloaded": False,  # Default
                "analyzed": False,  # Default
                "pre_page": None,
                "next_page": None,
                "meta_info": None
            }
            is_continuous = check_conversation_continuity(temp_last_page_in_batch, current_page_obj, self.client, model=self.llm_model)
            
            if is_continuous and temp_last_page_in_batch:
                # 如果前后两个page连续，则建立连接(pre、next)
                current_page_obj["pre_page"] = temp_last_page_in_batch["page_id"]
                last_meta = temp_last_page_in_batch.get("meta_info")
                # 用两个page一起更新meta信息
                new_meta = generate_page_meta_info(last_meta, current_page_obj, self.client, model=self.llm_model)
                current_page_obj["meta_info"] = new_meta
                if temp_last_page_in_batch.get("page_id") and self.mid_term_memory.get_page_by_id(temp_last_page_in_batch["page_id"]):

                    self._update_linked_pages_meta_info(temp_last_page_in_batch["page_id"], new_meta)
            else:

                current_page_obj["meta_info"] = generate_page_meta_info(None, current_page_obj, self.client, model=self.llm_model)
            
            current_batch_pages.append(current_page_obj)
            temp_last_page_in_batch = current_page_obj
        if current_batch_pages:  # 不为空

            self.last_evicted_page_for_continuity = current_batch_pages[-1]

        # Consolidate的意思是合并
        if not current_batch_pages:
            return

        input_text_for_summary = "\n".join([
            f"User: {p.get('user_input','')}\nAssistant: {p.get('agent_response','')}" 
            for p in current_batch_pages
        ])
        
        print("更新: 给被淘汰的batch生成multi-topic摘要...")
        multi_summary_result = gpt_generate_multi_summary(input_text_for_summary, self.client, model=self.llm_model)

        # 根据摘要的主题把驱逐的page插入到MTM
        if multi_summary_result and multi_summary_result.get("summaries"):
            for summary_item in multi_summary_result["summaries"]:
                # 摘要和关键词
                theme_summary = summary_item.get("content", "General summary of recent interactions.")
                theme_keywords = summary_item.get("keywords", [])
                print(f"更新: 处理主题 '{summary_item.get('theme')}' 为了插入MTM.")

                self.mid_term_memory.insert_pages_into_session(
                    summary_for_new_pages=theme_summary,
                    keywords_for_new_pages=theme_keywords,
                    pages_to_insert=current_batch_pages, # These pages now have pre_page, next_page, meta_info set up
                    similarity_threshold=self.topic_similarity_threshold
                )
        else:

            print("更新: multi-summary中没有特定主题. 把这些batch当作一个session添加.")
            fallback_summary = "General conversation segment from short-term memory."
            fallback_keywords = []  # Use empty keywords since multi-summary failed
            self.mid_term_memory.insert_pages_into_session(
                summary_for_new_pages=fallback_summary,
                keywords_for_new_pages=list(fallback_keywords),
                pages_to_insert=current_batch_pages,
                similarity_threshold=self.topic_similarity_threshold
            )

        for page in current_batch_pages:
            if page.get("pre_page"):
                # 更新pages间的前后连接
                self.mid_term_memory.update_page_connections(page["pre_page"], page["page_id"])
            if page.get("next_page"):
                 self.mid_term_memory.update_page_connections(page["page_id"], page["next_page"]) # This seems redundant if next is set by prior
        if current_batch_pages:  # Save if any pages were processed
            self.mid_term_memory.save()

    def update_long_term_from_analysis(self, user_id, profile_analysis_result):
        # 根据画像分析更新长期记忆
        if not profile_analysis_result:
            print("更新: 对于LTM更新，没有分析结果可用.")
            return
        # 画像
        new_profile_text = profile_analysis_result.get("profile")
        if new_profile_text and new_profile_text.lower() != "none":
            print(f"更新: 在LTM中给 {user_id} 更新用户画像.")
            # 直接使用新的分析结果作为完整画像，因为它应该已经是集成后的结果
            self.long_term_memory.update_user_profile(user_id, new_profile_text, merge=False)
        # 隐私
        user_private_knowledge = profile_analysis_result.get("private")
        if user_private_knowledge and user_private_knowledge.lower() != "none":
            print(f"更新: 在LTM中添加用户的隐私信息，user_id是 {user_id}.")
            # Split if multiple lines, assuming each line is a distinct piece of knowledge
            for line in user_private_knowledge.split('\n'):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_user_knowledge(line.strip()) 
        # 助手
        assistant_knowledge_text = profile_analysis_result.get("assistant_knowledge")
        if assistant_knowledge_text and assistant_knowledge_text.lower() != "none":
            print("更新: 在LTM中添加AI助手的知识.")
            for line in assistant_knowledge_text.split('\n'):
                if line.strip() and line.strip().lower() not in ["none", "- none", "- none."]:
                    self.long_term_memory.add_assistant_knowledge(line.strip())
