"""故事库 RAG 向量索引服务（FAISS）。

职责：
- 应用启动时从 SQLite stories 表全量构建向量索引（切块 → 嵌入 → FAISS）；
- 索引落盘到 data/story_faiss_index/，故事数或嵌入类型变化时自动重建；
- 提供语义检索 retrieve() 与单篇增量入索引 add_story()。

设计原则（课程 Day11）：嵌入对象与索引必须配套（同一套嵌入算法），
因此加载索引时使用进程内同一个 embeddings 实例；索引构建失败不阻断应用启动。
"""

from __future__ import annotations

import json
import threading
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.config import (
    ASSISTANT_INDEX_DIR,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
)
from backend.models import Story
from backend.services.ai_factory import embedding_tag, get_embeddings

# FAISS 兼容导入：优先独立包，回退社区包（与课程 Day11 一致）
try:  # pragma: no cover - 取决于安装的包
    from langchain_faiss import FAISS  # type: ignore
except ImportError:
    from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 中文友好的递归切分器：优先在段落/句子边界切，尽量不破坏语义
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)


# 模块级单例与并发锁
_lock = threading.Lock()
_vectorstore: Optional[Any] = None
_embeddings: Any = None
_meta: dict = {}
_meta_path = ASSISTANT_INDEX_DIR / "meta.json"
# 初始化是否已失败过；失败后不再重复尝试重建，避免每次 add_story 都打一次失败的嵌入 API
_init_failed: bool = False


def _story_to_documents(story: Story) -> list[Document]:
    """把一篇故事包装成 LangChain Document 并切块。"""
    doc = Document(
        page_content=f"《{story.title}》\n{story.content}",
        metadata={
            "story_id": story.id,
            "title": story.title,
            "category": story.category,
        },
    )
    return _splitter.split_documents([doc])


def _db_stats(db: Session) -> tuple[int, int]:
    """返回故事库当前 (故事数, 最大故事ID)。"""
    count = db.scalar(select(func.count(Story.id))) or 0
    max_id = db.scalar(select(func.max(Story.id))) or 0
    return int(count), int(max_id)


def _load_meta() -> dict:
    """读取落盘的索引 meta；不存在或损坏时返回空字典。"""
    try:
        if _meta_path.exists():
            return json.loads(_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[RAG索引] meta 读取失败，将重建索引：{exc}")
    return {}


def _save_meta(meta: dict) -> None:
    """把索引 meta 落盘。"""
    ASSISTANT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_index(db: Session) -> Optional[Any]:
    """从故事库全量构建 FAISS 索引并落盘。"""
    global _vectorstore, _embeddings, _meta

    stories = list(db.scalars(select(Story).order_by(Story.id)))
    if not stories:
        print("[RAG索引] 故事库为空，跳过向量索引构建（检索将返回空结果）")
        _meta = {"story_count": 0, "max_story_id": 0, "embedding": embedding_tag(_embeddings)}
        return None

    chunks: list[Document] = []
    for story in stories:
        chunks.extend(_story_to_documents(story))

    _vectorstore = FAISS.from_documents(chunks, _embeddings)

    # 落盘索引（FAISS 使用 pickle 序列化，本地自建索引可安全反序列化）
    ASSISTANT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(ASSISTANT_INDEX_DIR))

    count, max_id = _db_stats(db)
    _meta = {
        "story_count": count,
        "max_story_id": max_id,
        "chunk_count": len(chunks),
        "embedding": embedding_tag(_embeddings),
    }
    _save_meta(_meta)
    print(f"[RAG索引] 构建完成：{count} 篇故事 → {len(chunks)} 个文本块，已保存到 {ASSISTANT_INDEX_DIR}")
    return _vectorstore


def _load_index(db: Session) -> Optional[Any]:
    """从磁盘加载索引；meta 与当前故事库不一致时自动重建。"""
    global _vectorstore, _embeddings, _meta

    disk_meta = _load_meta()
    count, max_id = _db_stats(db)
    current_tag = embedding_tag(_embeddings)

    index_faiss = ASSISTANT_INDEX_DIR / "index.faiss"
    index_pkl = ASSISTANT_INDEX_DIR / "index.pkl"
    fresh = (
        disk_meta
        and disk_meta.get("story_count") == count
        and disk_meta.get("max_story_id") == max_id
        and disk_meta.get("embedding") == current_tag
        and index_faiss.exists()
        and index_pkl.exists()
    )
    if fresh:
        _vectorstore = FAISS.load_local(
            str(ASSISTANT_INDEX_DIR),
            _embeddings,
            allow_dangerous_deserialization=True,
        )
        _meta = disk_meta
        print(f"[RAG索引] 已加载本地索引（{count} 篇故事，嵌入类型 {current_tag}）")
        return _vectorstore

    if disk_meta:
        print("[RAG索引] 故事库或嵌入配置已变化，重建向量索引……")
    return _build_index(db)


def ensure_vector_index(db: Optional[Session] = None) -> Optional[Any]:
    """确保向量索引已就绪（进程内单例）。任何异常都降级为 None，不阻断调用方。"""
    global _vectorstore, _embeddings, _init_failed

    if _vectorstore is not None:
        return _vectorstore

    # 初始化已失败过则直接返回 None，避免重复调用失败的嵌入 API
    if _init_failed:
        return None

    with _lock:
        if _vectorstore is not None:
            return _vectorstore
        if _init_failed:
            return None
        own_session = False
        if db is None:
            from backend.database import SessionLocal

            db = SessionLocal()
            own_session = True
        try:
            _embeddings = get_embeddings()
            return _load_index(db)
        except Exception as exc:
            # 启动健壮性：索引失败不影响应用其余功能
            print(f"[RAG索引] 初始化失败，语义检索暂不可用：{exc}")
            _vectorstore = None
            _init_failed = True
            return None
        finally:
            if own_session:
                db.close()


def retrieve(query: str, k: Optional[int] = None) -> list[dict]:
    """语义检索：返回最相关的 k 个文本块（含出处元数据）。"""
    store = ensure_vector_index()
    if store is None or not query.strip():
        return []
    try:
        docs = store.similarity_search(query, k=k or RAG_TOP_K)
    except Exception as exc:
        print(f"[RAG索引] 检索失败：{exc}")
        return []
    results = []
    for doc in docs:
        results.append(
            {
                "story_id": doc.metadata.get("story_id"),
                "title": doc.metadata.get("title", "未知故事"),
                "category": doc.metadata.get("category", ""),
                "content": doc.page_content,
            }
        )
    return results


def add_story(story: Story) -> bool:
    """新故事增量加入向量索引并落盘；返回是否成功。"""
    store = ensure_vector_index()
    if store is None:
        # 索引尚未初始化（可能故事库原本为空）：尝试全量重建
        from backend.database import SessionLocal

        with SessionLocal() as db:
            store = ensure_vector_index(db)
        if store is None:
            return False
    try:
        with _lock:
            chunks = _story_to_documents(story)
            store.add_documents(chunks)
            store.save_local(str(ASSISTANT_INDEX_DIR))
            _meta["story_count"] = int(_meta.get("story_count", 0)) + 1
            _meta["max_story_id"] = max(int(_meta.get("max_story_id", 0)), story.id)
            _meta["chunk_count"] = int(_meta.get("chunk_count", 0)) + len(chunks)
            _save_meta(_meta)
        print(f"[RAG索引] 新故事《{story.title}》已增量入索引（{len(chunks)} 块）")
        return True
    except Exception as exc:
        print(f"[RAG索引] 增量写入失败（不影响本次创作）：{exc}")
        return False
