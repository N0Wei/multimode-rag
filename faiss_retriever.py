#!/usr/bin/env python
# coding: utf-8

# LangChain 1.0: Document 在 langchain-core 中
from langchain_core.documents import Document
# LangChain 1.0: VectorStores 移到了 langchain-community
from langchain_community.vectorstores import FAISS
from pdf_parse import DataProcess
from embedding_model import get_embedding_model  # 使用统一的嵌入模型接口
from config import EMBEDDING_BACKEND, EMBEDDING_MODEL_PATH, OLLAMA_EMBEDDING_MODEL, EMBEDDING_MODEL_NAME
import torch
import hashlib
import pickle
from pathlib import Path


class FaissRetriever(object):
    """
    FAISS 检索器 + 本地缓存
    
    优化:
    - 首次运行: 构建索引 + 保存到本地
    - 后续运行: 从本地加载（秒级启动）
    - 数据变化: 自动检测并重建
    - 支持本地/API 嵌入模型切换
    - 模型变化: 自动检测并重建索引
    """
    
    def __init__(
        self, 
        model_path=None,  # 可选，如果提供则强制使用本地模型
        data=None,
        cache_dir: str = "./vector_cache",
        force_rebuild: bool = False
    ):
        """
        Args:
            model_path: 嵌入模型路径（可选，如果为 None 则从配置读取）
            data: 文档数据列表
            cache_dir: 缓存目录
            force_rebuild: 是否强制重建索引
        """
        self.model_path = model_path  # 保留用于兼容性
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算数据指纹（包含模型信息）
        self.data_hash = self._calculate_data_hash(data)
        # FAISS 会生成 .faiss 和 .pkl 两个文件，使用 index_name（不含扩展名）
        self.index_name = f"faiss_index_{self.data_hash}"
        self.cache_faiss_path = self.cache_dir / f"{self.index_name}.faiss"
        self.cache_pkl_path = self.cache_dir / f"{self.index_name}.pkl"
        self.metadata_path = self.cache_dir / f"faiss_metadata_{self.data_hash}.pkl"
        
        # 尝试从缓存加载
        if not force_rebuild and self._load_from_cache():
            print(f"  ✓ FAISS 索引已从缓存加载: {self.cache_faiss_path.name}")
        else:
            # 缓存不存在或数据变化，重新构建
            print(f"  🔨 构建 FAISS 索引...")
            self._build_index(data)
            self._save_to_cache()
            print(f"  ✓ FAISS 索引已构建并缓存: {self.cache_faiss_path.name}")
        
        # 释放显存
        if hasattr(self, 'embeddings') and hasattr(self.embeddings, 'client'):
            # 如果是本地模型，释放显存
            try:
                del self.embeddings
                torch.cuda.empty_cache()
            except:
                pass
    
    def _calculate_data_hash(self, data) -> str:
        """
        计算数据指纹
        用于检测数据是否变化
        同时包含模型信息，确保模型切换时重建索引
        """
        # 获取模型标识符
        if self.model_path:
            # 如果指定了 model_path，使用它作为标识
            model_identifier = f"local:{self.model_path}"
        else:
            # 使用配置中的模型信息
            backend = EMBEDDING_BACKEND.lower()
            if backend == "ollama":
                model_identifier = f"ollama:{OLLAMA_EMBEDDING_MODEL}"
            elif backend == "api":
                # API 模式需要包含 API 类型，因为不同 API 的向量空间不同
                from config import EMBEDDING_API_TYPE
                api_type = EMBEDDING_API_TYPE.lower()
                model_identifier = f"api:{api_type}:{EMBEDDING_MODEL_NAME}"
            else:
                model_identifier = f"local:{EMBEDDING_MODEL_PATH}"
        
        # 使用前100个文档 + 总数量 + 模型标识 计算 hash
        sample_data = data[:100] if len(data) > 100 else data
        data_str = str(len(data)) + "".join(sample_data[:10]) + model_identifier
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _build_index(self, data):
        """构建 FAISS 索引（支持分批处理）"""
        # 使用统一的嵌入模型接口
        if self.model_path:
            # 如果提供了路径，强制使用本地模型（向后兼容）
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from config import EMBEDDING_DEVICE
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={"device": EMBEDDING_DEVICE}
            )
        else:
            # 使用配置的嵌入模型
            self.embeddings = get_embedding_model()
        
        # 准备文档
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        
        # 根据不同 API 设置批次大小
        # OpenAI: token 限制（每个文档平均约 100-500 tokens），保守设置
        # Jina AI: 文档数量限制（最多 512 个）
        # 本地模型: 根据显存设置
        backend = EMBEDDING_BACKEND.lower()
        if backend == "api":
            from config import EMBEDDING_API_TYPE
            api_type = EMBEDDING_API_TYPE.lower()
            if api_type == "jina":
                batch_size = 500  # Jina 限制 512，留一点余量
            else:  # openai 或其他
                batch_size = 100  # OpenAI 有 token 限制，保守设置
        else:
            batch_size = 500  # 本地模型或 Ollama
        
        total_docs = len(docs)
        
        if total_docs > batch_size:
            # 分批构建索引
            print(f"  📦 文档数量: {total_docs}，将分 {(total_docs + batch_size - 1) // batch_size} 批处理（每批 {batch_size} 个文档）")
            
            # 第一批：创建初始索引
            first_batch = docs[:batch_size]
            print(f"  ⏳ 处理第 1 批 ({len(first_batch)} 个文档)...")
            self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
            
            # 后续批次：添加到现有索引
            for i in range(batch_size, total_docs, batch_size):
                batch_num = (i // batch_size) + 1
                batch = docs[i:i + batch_size]
                print(f"  ⏳ 处理第 {batch_num} 批 ({len(batch)} 个文档)...")
                
                # 添加到现有索引
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                self.vector_store.add_texts(texts, metadatas=metadatas)
        else:
            # 文档数量少，直接构建
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
    
    def _save_to_cache(self):
        """保存索引到本地"""
        try:
            # 保存 FAISS 索引（会生成 .faiss 和 .pkl 两个文件）
            self.vector_store.save_local(
                str(self.cache_dir), 
                index_name=self.index_name
            )
            
            # 保存元数据（包含模型信息）
            backend = EMBEDDING_BACKEND.lower()
            if self.model_path:
                model_info = f"local:{self.model_path}"
            elif backend == "ollama":
                model_info = f"ollama:{OLLAMA_EMBEDDING_MODEL}"
            elif backend == "api":
                from config import EMBEDDING_API_TYPE
                api_type = EMBEDDING_API_TYPE.lower()
                model_info = f"api:{api_type}:{EMBEDDING_MODEL_NAME}"
            else:
                model_info = f"local:{EMBEDDING_MODEL_PATH}"
            
            metadata = {
                "data_hash": self.data_hash,
                "model_info": model_info,
                "embedding_backend": backend,
                "created_at": str(Path(__file__).stat().st_mtime)
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"  💾 缓存已保存 [模型: {model_info}]")
            
            # 清理旧缓存
            self._cleanup_old_caches()
            
        except Exception as e:
            print(f"  ⚠️  缓存保存失败: {e}")
    
    def _load_from_cache(self) -> bool:
        """从本地加载索引"""
        try:
            # 检查缓存文件是否存在（FAISS 需要 .faiss 和 .pkl 两个文件）
            if not self.cache_faiss_path.exists() or not self.cache_pkl_path.exists():
                print(f"  ℹ️  缓存文件不存在，将构建新索引")
                return False
            
            # 读取元数据验证模型匹配
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        model_info = metadata.get("model_info", "unknown")
                        print(f"  📂 发现缓存 [模型: {model_info}]")
                except:
                    pass
            
            # 加载嵌入模型（需要用于查询）
            if self.model_path:
                # 如果提供了路径，强制使用本地模型（向后兼容）
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from config import EMBEDDING_DEVICE
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_path,
                    model_kwargs={"device": EMBEDDING_DEVICE}
                )
            else:
                # 使用配置的嵌入模型
                self.embeddings = get_embedding_model()
            
            # 加载 FAISS 索引
            self.vector_store = FAISS.load_local(
                str(self.cache_dir),
                self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True  # 信任本地文件
            )
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  缓存加载失败: {e}")
            return False
    
    def _cleanup_old_caches(self):
        """清理旧的缓存文件"""
        try:
            # 保留当前缓存，删除其他
            for file in self.cache_dir.glob("faiss_*"):
                if not file.name.startswith(f"faiss_index_{self.data_hash}") and \
                   not file.name.startswith(f"faiss_metadata_{self.data_hash}"):
                    file.unlink()
                    print(f"  🗑️  清理旧缓存: {file.name}")
        except Exception as e:
            print(f"  ⚠️  清理缓存失败: {e}")

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store

if __name__ == "__main__":
    base = "."
    # model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    model_name = ""
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
