import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

import warnings

try:
    # Suppress SWIG deprecation warnings from FAISS
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
    
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)

    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
        "or `pip install faiss-cpu` (depending on Python version)."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]  # memory id
    score: Optional[float]  # distance
    payload: Optional[Dict]  # metadata


class FAISS(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        path: Optional[str] = None,
        distance_strategy: str = "euclidean",
        normalize_L2: bool = False,
        embedding_model_dims: int = 1536,
    ):
        """
        Initialize the FAISS vector store.
        Args:
            collection_name (str): Name of the collection.
            path (str, optional): Path for local FAISS database. Defaults to None.
            distance_strategy (str, optional): Distance strategy to use. Options: 'euclidean', 'inner_product', 'cosine'.
                Defaults to "euclidean".
            normalize_L2 (bool, optional): Whether to normalize L2 vectors. Only applicable for euclidean distance.
                Defaults to False.
        """
        self.collection_name = collection_name
        self.path = path or f"/tmp/faiss/{collection_name}"
        self.distance_strategy = distance_strategy
        self.normalize_L2 = normalize_L2
        self.embedding_model_dims = embedding_model_dims

        # Initialize storage structures
        self.index = None
        self.docstore = {}
        self.index_to_id = {}
        # -------------------------------- 添加反向映射
        self.id_to_index = {}
        self.deleted_indices = set()
        # --------------------------------

        # Create directory if it doesn't exist
        # if self.path:
        #     os.makedirs(os.path.dirname(self.path), exist_ok=True)
        #     # Try to load existing index if available
        #     index_path = f"{self.path}/{collection_name}.faiss"
        #     docstore_path = f"{self.path}/{collection_name}.pkl"
        #     if os.path.exists(index_path) and os.path.exists(docstore_path):
        #         self._load(index_path, docstore_path)
        #     else:
        #         self.create_col(collection_name)

        # ----------------------------
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            # Try to load existing index if available
            index_path = f"{self.path}/{collection_name}.faiss"
            docstore_path = f"{self.path}/{collection_name}.pkl"
            if os.path.exists(index_path) and os.path.exists(docstore_path):
                self._load(index_path, docstore_path)
            else:
                self.create_col(collection_name)
        # ---------------------------

    def _load(self, index_path: str, docstore_path: str):
        """
        Load FAISS index and docstore from disk.

        Args:
            index_path (str): Path to FAISS index file.
            docstore_path (str): Path to docstore pickle file.
        """
        # try:
        #     self.index = faiss.read_index(index_path)
        #     with open(docstore_path, "rb") as f:
        #         self.docstore, self.index_to_id = pickle.load(f)
        #     logger.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} vectors")
        # except Exception as e:
        #     logger.warning(f"Failed to load FAISS index: {e}")
        #     self.docstore = {}
        #     self.index_to_id = {}
        # ---------------------------------
        try:
            self.index = faiss.read_index(index_path)
            with open(docstore_path, "rb") as f:
                data = pickle.load(f)
            # 向后兼容：检查是旧格式还是新格式
            if isinstance(data, tuple) and len(data) == 2:
                # 旧格式：(docstore, index_to_id)
                self.docstore, self.index_to_id = data
                self.id_to_index = {}
                self.deleted_indices = set()
                # 重建id_to_index
                for idx, vector_id in self.index_to_id.items():
                    self.id_to_index[vector_id] = idx
            else:
                # 新格式：字典
                self.docstore = data.get('docstore', {})
                self.index_to_id = data.get('index_to_id', {})
                self.id_to_index = data.get('id_to_index', {})
                deleted_indices_list = data.get('deleted_indices', [])
                self.deleted_indices = set(deleted_indices_list)
            logger.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self.docstore = {}
            self.index_to_id = {}
            self.id_to_index = {}
            self.deleted_indices = set()
        # ---------------------------------

    def _save(self):
        """Save FAISS index and docstore to disk."""
        if not self.path or not self.index:
            return
        try:
            os.makedirs(self.path, exist_ok=True)
            index_path = f"{self.path}/{self.collection_name}.faiss"
            docstore_path = f"{self.path}/{self.collection_name}.pkl"

            faiss.write_index(self.index, index_path)
            # ------------------------------
            # 保存所有需要持久化的数据
            data_to_save = {
                'docstore': self.docstore,
                'index_to_id': self.index_to_id,
                'id_to_index': getattr(self, 'id_to_index', {}),  # 保存id_to_index
                'deleted_indices': list(getattr(self, 'deleted_indices', set())),  # 保存已删除索引
            }
            # -----------------------------
            # with open(docstore_path, "wb") as f:
            #     pickle.dump((self.docstore, self.index_to_id), f)
            with open(docstore_path, "wb") as f:
                pickle.dump(data_to_save, f)
        except Exception as e:
            logger.warning(f"Failed to save FAISS index: {e}")

    def _parse_output(self, scores, ids, limit=None) -> List[OutputData]:
        """
        Parse the output data.
        Args:
            scores: Similarity scores from FAISS.
            ids: Indices from FAISS.
            limit: Maximum number of results to return.
        Returns:
            List[OutputData]: Parsed output data.
        """
        print(f"🔧 _parse_output调试:")
        print(f"  输入: ids={ids[:5]}... (共{len(ids)}个), scores={scores[:5]}...")
        print(f"  index_to_id长度: {len(self.index_to_id) if hasattr(self, 'index_to_id') else '无此属性'}")
        print(f"  docstore类型: {type(self.docstore) if hasattr(self, 'docstore') else '无此属性'}")
        if limit is None:
            limit = len(ids)
        results = []
        for i in range(min(len(ids), limit)):
            if ids[i] == -1:  # FAISS returns -1 for empty results
                continue
            index_id = int(ids[i])
            print(f"  处理 index_id={index_id}")
            # ----------------------------- 检查是否在已删除的索引中
            if index_id in self.deleted_indices:  # 直接使用，不需要hasattr检查
                continue
            # -----------------------------
            vector_id = self.index_to_id.get(index_id)
            print(f"    映射到的vector_id: {vector_id}")
            if vector_id is None:
                print(f"    ❌ 映射失败! index_id {index_id} 不在 index_to_id 中")
                print(f"    index_to_id内容示例: {list(self.index_to_id.items())[:3] if self.index_to_id else '空'}")
                continue
            payload = self.docstore.get(vector_id)
            print(f"    从docstore获取payload: {payload is not None}")
            if payload is None:
                print(f"    ❌ docstore中没有vector_id={vector_id}")
                continue
            payload_copy = payload.copy()
            score = float(scores[i])
            entry = OutputData(
                id=vector_id,
                score=score,
                payload=payload_copy,
            )
            results.append(entry)
            print(f"    ✅ 添加结果 {len(results)}: id={vector_id}, score={score}")
        print(f"  最终返回 {len(results)} 个结果")
        return results

    def create_col(self, name: str, distance: str = None):
        """
        Create a new collection.

        Args:
            name (str): Name of the collection.
            distance (str, optional): Distance metric to use. Overrides the distance_strategy
                passed during initialization. Defaults to None.

        Returns:
            self: The FAISS instance.
        """
        distance_strategy = distance or self.distance_strategy

        # Create index based on distance strategy
        if distance_strategy.lower() == "inner_product" or distance_strategy.lower() == "cosine":
            self.index = faiss.IndexFlatIP(self.embedding_model_dims)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_model_dims)

        self.collection_name = name

        self._save()

        return self

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Insert vectors into a collection.
        Args:
            vectors (List[list]): List of vectors to insert.
            payloads (Optional[List[Dict]], optional): List of payloads corresponding to vectors. Defaults to None.
            ids (Optional[List[str]], optional): List of IDs corresponding to vectors. Defaults to None.
        """
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            raise ValueError("Vectors, payloads, and IDs must have the same length")
        # ------------------------------ 检查ID是否重复
        # 🔴 重要：检查ID是否重复
        duplicate_ids = []
        for vector_id in ids:
            if vector_id in self.id_to_index:
                duplicate_ids.append(vector_id)
        if duplicate_ids:
            raise ValueError(f"Duplicate vector IDs found: {duplicate_ids}")
        # ------------------------------
        vectors_np = np.array(vectors, dtype=np.float32)
        if self.normalize_L2 and self.distance_strategy.lower() == "euclidean":
            faiss.normalize_L2(vectors_np)

        # -------------------------------
        # self.index.add(vectors_np)
        # starting_idx = len(self.index_to_id)
        starting_idx = self.index.ntotal
        self.index.add(vectors_np)
        # ----------------------------------

        for i, (vector_id, payload) in enumerate(zip(ids, payloads)):
            self.docstore[vector_id] = payload.copy()
            self.index_to_id[starting_idx + i] = vector_id
            # --------------------------------
            self.id_to_index[vector_id] = starting_idx + i
            # --------------------------------
        self._save()
        logger.info(f"Inserted {len(vectors)} vectors into collection {self.collection_name}")

    def search(
        self, query: str, vectors: List[list], limit: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.
        Args:
            query (str): Query (not used, kept for API compatibility).
            vectors (List[list]): List of vectors to search.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (Optional[Dict], optional): Filters to apply to the search. Defaults to None.
        Returns:
            List[OutputData]: Search results.
        """
        print(f"🔍 FAISS搜索调试:")
        print(f"  索引状态: {'已初始化' if self.index else '未初始化'}")

        total_in_faiss = self.index.ntotal
        total_mappings = len(self.index_to_id)
        deleted_count = len(self.deleted_indices)
        active_count = total_mappings  # 活跃向量数 = 映射数
        print(f"FAISS总向量数: {total_in_faiss}")
        print(f"活跃向量数（有映射的）: {active_count}")
        print(f"已标记删除数: {deleted_count}")
        print(f"幽灵向量数（无映射的）: {total_in_faiss - active_count - deleted_count}")

        if self.index is not None:
            print(f"  索引向量数: {self.index.ntotal}")
            print(f"  索引维度: {self.index.d}")
        print(f"  查询向量形状: {np.array(vectors).shape}")
        print(f"  filters: {filters}")
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")
        query_vectors = np.array(vectors, dtype=np.float32)
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if self.normalize_L2 and self.distance_strategy.lower() == "euclidean":
            faiss.normalize_L2(query_vectors)
        fetch_k = limit * 2 if filters else limit
        scores, indices = self.index.search(query_vectors, fetch_k)
        # 在搜索后添加调试
        print(f"  搜索返回: indices={indices[0]}, scores={scores[0]}")
        results = self._parse_output(scores[0], indices[0], limit)
        print(f"  解析后结果数: {len(results)}")
        if filters:
            print("有filters")
            filtered_results = []
            for result in results:
                if self._apply_filters(result.payload, filters):
                    filtered_results.append(result)
                    if len(filtered_results) >= limit:
                        break
            results = filtered_results[:limit]
        return results

    def _apply_filters(self, payload: Dict, filters: Dict) -> bool:
        """
        Apply filters to a payload.
        Args:
            payload (Dict): Payload to filter.
            filters (Dict): Filters to apply.
        Returns:
            bool: True if payload passes filters, False otherwise.
        """
        if not filters or not payload:
            return True
        for key, value in filters.items():
            if key not in payload:
                return False
            if isinstance(value, list):
                if payload[key] not in value:
                    return False
            elif payload[key] != value:
                return False
        return True

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.
        Args:
            vector_id (str): ID of the vector to delete.
        """
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")
        # index_to_delete = None
        # for idx, vid in self.index_to_id.items():
        #     if vid == vector_id:
        #         index_to_delete = idx
        #         break
        # ------------------------------------
        index_to_delete = self.id_to_index.get(vector_id)
        # -----------------------------------
        if index_to_delete is not None:
            # self.docstore.pop(vector_id, None)
            # self.index_to_id.pop(index_to_delete, None)
            # self._save()
            # logger.info(f"Deleted vector {vector_id} from collection {self.collection_name}")

            self.deleted_indices.add(index_to_delete)
            # 更新映射关系
            # 更新映射关系
            if index_to_delete in self.index_to_id:
                del self.index_to_id[index_to_delete]
            if vector_id in self.id_to_index:
                del self.id_to_index[vector_id]
            if vector_id in self.docstore:
                del self.docstore[vector_id]
            self._save()
            logger.info(f"Deleted vector {vector_id} from collection {self.collection_name}")
            # -------------------------------
        else:
            logger.warning(f"Vector {vector_id} not found in collection {self.collection_name}")

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ):
        """
        Update a vector and its payload.

        Args:
            vector_id (str): ID of the vector to update.
            vector (Optional[List[float]], optional): Updated vector. Defaults to None.
            payload (Optional[Dict], optional): Updated payload. Defaults to None.
        """
        # ---------------------------------
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")
        if vector_id not in self.docstore:
            raise ValueError(f"Vector {vector_id} not found")
        current_payload = self.docstore[vector_id].copy()
        if payload is not None:
            self.docstore[vector_id] = payload.copy()
            current_payload = self.docstore[vector_id].copy()
        if vector is not None:
            self.delete(vector_id)
            self.insert([vector], [current_payload], [vector_id])
        else:
            self._save()
        logger.info(f"Updated vector {vector_id} in collection {self.collection_name}")
        # ------------------------------------------
        # if self.index is None:
        #     raise ValueError("Collection not initialized. Call create_col first.")
        # if vector_id not in self.docstore:
        #     raise ValueError(f"Vector {vector_id} not found")
        #     # 🔧 修复：处理vector参数
        # # 找到当前位置
        # current_position = None
        # for pos, vid in self.index_to_id.items():
        #     if vid == vector_id:
        #         current_position = pos
        #         break
        #
        # if vector is not None and current_position is not None:
        #     import numpy as np
        #     vector_np = np.asarray(vector, dtype=np.float32)
        #     if len(vector_np.shape) == 1:
        #         vector_np = vector_np.reshape(1, -1)
        #     print(f"🔧 UPDATE: 形状={vector_np.shape}")
        #     # 归一化
        #     if self.normalize_L2 and self.distance_strategy.lower() == "euclidean":
        #         faiss.normalize_L2(vector_np)
        #     # 1. 从FAISS删除旧向量
        #     self.index.remove_ids([current_position])
        #     # 2. 添加新向量（会在末尾）
        #     self.index.add(vector_np)
        #     # 3. 新向量在self.index.ntotal-1位置
        #     new_position = self.index.ntotal - 1
        #     # 4. 关键：更新映射，删除旧映射！
        #     self.index_to_id.pop(current_position, None)  # 删除旧位置
        #     self.index_to_id[new_position] = vector_id  # 添加新位置
        # # 🔧 修复2：只更新payload的情况
        # elif vector is None and current_position is not None:
        #     # 只更新payload，向量不变，映射也不变
        #     pass
        # # 更新payload
        # if payload is not None:
        #     self.docstore[vector_id] = payload.copy()
        # self._save()

    def get(self, vector_id: str) -> OutputData:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str): ID of the vector to retrieve.

        Returns:
            OutputData: Retrieved vector.
        """
        if self.index is None:
            raise ValueError("Collection not initialized. Call create_col first.")

        if vector_id not in self.docstore:
            return None

        payload = self.docstore[vector_id].copy()

        return OutputData(
            id=vector_id,
            score=None,
            payload=payload,
        )

    def list_cols(self) -> List[str]:
        """
        List all collections.

        Returns:
            List[str]: List of collection names.
        """
        if not self.path:
            return [self.collection_name] if self.index else []

        try:
            collections = []
            path = Path(self.path).parent
            for file in path.glob("*.faiss"):
                collections.append(file.stem)
            return collections
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return [self.collection_name] if self.index else []

    def delete_col(self):
        """
        Delete a collection.
        """
        if self.path:
            try:
                index_path = f"{self.path}/{self.collection_name}.faiss"
                docstore_path = f"{self.path}/{self.collection_name}.pkl"

                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(docstore_path):
                    os.remove(docstore_path)

                logger.info(f"Deleted collection {self.collection_name}")
            except Exception as e:
                logger.warning(f"Failed to delete collection: {e}")

        self.index = None
        self.docstore = {}
        self.index_to_id = {}
        self.id_to_index = {}
        self.deleted_indices = set()

    def col_info(self) -> Dict:
        """
        Get information about a collection.

        Returns:
            Dict: Collection information.
        """
        if self.index is None:
            return {"name": self.collection_name, "count": 0}

        return {
            "name": self.collection_name,
            "count": self.index.ntotal,
            "dimension": self.index.d,
            "distance": self.distance_strategy,
        }

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[OutputData]:
        """
        List all vectors in a collection.

        Args:
            filters (Optional[Dict], optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            List[OutputData]: List of vectors.
        """
        if self.index is None:
            return []
        results = []
        count = 0
        for vector_id, payload in self.docstore.items():
            if filters and not self._apply_filters(payload, filters):
                continue
            payload_copy = payload.copy()
            results.append(
                OutputData(
                    id=vector_id,
                    score=None,
                    payload=payload_copy,
                )
            )
            count += 1
            if count >= limit:
                break
        return [results]

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.collection_name)
