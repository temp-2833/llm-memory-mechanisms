import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError
# 导入mem0的内部模块
from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)

# 全局忽略SWIG弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# 日志记录器
logger = logging.getLogger(__name__)


def _safe_deepcopy_config(config):
    # 安全地深拷贝，对于不可序列化的对象回退到JSON序列化
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"深拷贝失败，使用JSON序列化: {e}")
        
        config_class = type(config)
        
        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump(mode="json")
            except Exception:
                clone_dict = {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            clone_dict = asdict(config)
        else:
            clone_dict = {k: v for k, v in config.__dict__.items()}
        
        sensitive_tokens = ("auth", "credential", "password", "token", "secret", "key", "connection_class")
        for field_name in list(clone_dict.keys()):
            if any(token in field_name.lower() for token in sensitive_tokens):
                clone_dict[field_name] = None
        
        try:
            return config_class(**clone_dict)
        except Exception as reconstruction_error:
            logger.warning(
                f"Failed to reconstruct config: {reconstruction_error}. "
                f"Telemetry may be affected."
            )
            raise


def _build_filters_and_metadata(
    *,  # 强制关键字参数
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    # 基于session和参与者标识符，构建存储的metadata和查询筛选器filter
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id: 用户标识符
        agent_id: 代理标识符
        run_id: 运行标识符
        actor_id: 显式参与者标识符，用于特定参与者筛选
        input_metadata: 用于存储元数据模板的基础元数据
        input_filters: 用于查询筛选器的基础筛选条件

    Returns:
        tuple: (base_metadata_template, effective_query_filters)
        - base_metadata_template: 存储新记忆的元数据模板
        - effective_query_filters: 查询现有记忆的筛选器
    """
    # 深拷贝输入以避免修改原始数据
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- 添加所有提供的session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- 可选的参与者筛选器 ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


# 配置和日志
setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    # 用于创建、检索、更新和删除记忆
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        # 自定义提示词
        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        # 组建embedding的各个组件
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        
        # 初始化重排序器（如果配置了）
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider, 
                config.reranker.config
            )
        # 初始化图存储
        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None
        # 为遥测创建配置，避免深度拷贝时的线程锁问题
        telemetry_config_dict = {}
        if hasattr(self.config.vector_store.config, 'model_dump'):
            # For pydantic models
            telemetry_config_dict = self.config.vector_store.config.model_dump()
        else:
            # For other objects, manually copy common attributes
            for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                if hasattr(self.config.vector_store.config, attr):
                    telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

        # 为遥测覆盖集合名称
        telemetry_config_dict['collection_name'] = "mem0migrations"

        # 设置基于文件的向量存储的路径
        telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
            os.makedirs(telemetry_config_dict['path'], exist_ok=True)

        # 使用与原始配置相同的类创建遥测配置对象
        telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, telemetry_config
        )
        # 记录初始化事件
        capture_event("mem0.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        # 从配置字典创建Memory实例
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        # 处理config-dict
        # 如果配置了图存储但未配置向量存储，则创建默认向量存储配置
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        # 根据逻辑判断是否使用代理记忆提取
        """
        - 如果metadata中存在agent_id且messages包含assistant角色 -> True
        - 否则 -> False
        Args:
            messages: 消息字典列表
            metadata: 包含user_id、agent_id等的元数据
        Returns:
            bool: True表示应使用代理记忆提取，False表示用户记忆提取
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        创建新记忆
        添加限定到单个会话ID的新记忆（如user_id、agent_id或run_id）.
        One of those ids is required.
        Args:
            infer: 如果为True，使用LLM从messages中提取关键事实并决定添加、更新或删除相关记忆
            memory_type: 指定记忆类型，目前只支持MemoryType.PROCEDURAL.value
            prompt: 用于记忆创建的提示词
        Returns:
            dict: 包含记忆添加操作结果的字典
        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """
        # 构建筛选器和元数据
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )
        # 验证memory type
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )
        # 如果是过程记忆且提供了agent_id
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results
        # 解析视觉消息(如果启用)
        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)
        # 使用线程池并行执行向量存储和图存储操作
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 这里，_add_to_vector_store
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)
            concurrent.futures.wait([future1, future2])
            vector_store_result = future1.result()
            graph_result = future2.result()
        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        # 将消息添加到向量存储
        if not infer:  # 如果不推断，直接添加原始消息
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue
                if message_dict["role"] == "system":
                    continue
                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]
                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name
                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)
                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # 推断模式：使用LLM提取关键事实
        parsed_messages = parse_messages(messages)
        # 使用自定义或默认提示词
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # Determine if this should use agent memory extraction based on agent_id presence
            # and role types in messages
            is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)
        # 调用llm
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        try:
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        # 搜索现有记忆，使用提供的 session identifiers
        # Use all available session identifiers for accurate memory retrieval
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]
        for new_mem in new_retrieved_facts:  # 把所有llm提取的facts进行处理
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
        # 去重
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # 为UUID创建临时映射以处理幻觉
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            function_calling_prompt = get_update_memory_messages(
                retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
            )
            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""
            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue
                    event_type = resp.get("event")
                    if event_type == "ADD":  # 增
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append({"id": memory_id, "memory": action_text, "event": event_type})
                    elif event_type == "UPDATE":  # 改
                        # ----------------------------
                        # old_text = resp.get("old_memory", "")
                        # if action_text == old_text:
                        #     print(f"⚠️ 跳过无用UPDATE: 内容相同 '{action_text[:50]}...'")
                        #     continue
                        # ---------------------------
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":  # 删
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Update only the session identifiers, keep content the same
                            existing_memory = self.vector_store.get(vector_id=memory_id)
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                            self.vector_store.update(
                                vector_id=memory_id,
                                vector=None,  # Keep same embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        # 将消息添加到图存储
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        # 通过ID检索记忆
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}
        # 创建MemoryItem并转换为字典
        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        # 列出所有记忆
        # 构建筛选器
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )
        # 并行获取向量存储和图存储数据
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        # 从向量存储获取所有记忆
        memories_result = self.vector_store.list(filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ):
        # 搜索记忆
        """
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
        """
        # 目前filter应该默认是None
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # 如果检测到高级运算符，处理元数据筛选器
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)
        # effective-filter应该是一个字典，指定key是user-id
        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )
        # 并行搜索向量存储和图存储
        with concurrent.futures.ThreadPoolExecutor() as executor:

            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # 如果启用了重排序且重排序器可用，则应用重排序
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        # 处理增强的元数据筛选器并将其转换为向量存储兼容格式
        processed_filters = {}
        # 处理单个条件
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}
            
            result = {}
            for operator, value in condition.items():
                # # 将平台运算符映射到通用格式
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte", 
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }
                
                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result
        # 处理不同类型的筛选器
        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))
        
        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        # 检查筛选器是否包含需要特殊处理的高级运算符
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        # 在向量存储中搜索
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)
        print(f"在Memory类中")
        # print("embeddings:")
        # print(embeddings)
        print("有filters过滤器，内容如下：")
        print(filters)
        print("memories:")
        print(memories)
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id", *promoted_payload_keys}
        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
            ).model_dump()
            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]
            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata
            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    def update(self, memory_id, data):
        # 通过ID更新记忆
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        # 通过ID删除记忆
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        # 删除所有记忆
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # 删除所有向量记忆并重置集合
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        self.vector_store.reset()

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        # 获取记忆的更改历史
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        # 创建新记忆
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        # 创建过程记忆
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        # 更新记忆
        logger.info(f"Updating memory with {data=}")
        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        # --------------------------------------
        # import numpy as np
        # if isinstance(existing_embeddings, dict):  # 如果是字典
        #     if data in existing_embeddings:
        #         embedding_list = existing_embeddings[data]
        #     else:
        #         embedding_list = self.embedding_model.embed(data, "update")
        # else:  # 如果不是字典
        #     embedding_list = existing_embeddings
        # # 转换为numpy数组
        # if isinstance(embedding_list, list):
        #     embedding_np = np.array(embedding_list, dtype=np.float32)
        # else:
        #     embedding_np = embedding_list
        # print(f"🔧 _update_memory调试:")
        # print(f"  文本: '{data[:50]}...'")
        # print(f"  向量类型: {type(embedding_np)}")
        # print(f"  形状: {embedding_np.shape if hasattr(embedding_np, 'shape') else '无'}")
        #
        # self.vector_store.update(
        #     vector_id=memory_id,
        #     vector=embedding_np,
        #     payload=new_metadata,
        # )
        # logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        # self.db.add_history(
        #     memory_id,
        #     prev_value,
        #     data,
        #     "UPDATE",
        #     created_at=new_metadata["created_at"],
        #     updated_at=new_metadata["updated_at"],
        #     actor_id=new_metadata.get("actor_id"),
        #     role=new_metadata.get("role"),
        # )
        # return memory_id


        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id):
        # 删除记忆
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    def reset(self):
        # 重置记忆存储：1.删除向量存储；2.重置数据库；3.使用新client重新创建向量存储
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    def chat(self, query):
        raise NotImplementedError("聊天功能尚未实现.")
