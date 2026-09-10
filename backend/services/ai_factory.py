"""LangChain 模型工厂：聊天模型与嵌入模型的双模式（真模型 / 教学模拟）统一入口。

- 配置了 TINYLLM_TEXT_API_*（OpenAI 兼容对话接口）→ 使用 langchain_openai.ChatOpenAI；
- 配置了 TINYLLM_EMBED_*（支持 embedding 的接口）→ 使用 OpenAIEmbeddings；
- 未配置或调用失败 → 回退到内置 CharEmbeddings（字符 n-gram 哈希教学嵌入），
  保证无 Key、离线环境下 RAG 全流程依然可运行。

教学嵌入原理（课程 Day11）：把文本切成 1/2/3 字片段（n-gram），
每个片段经 md5 哈希映射到固定维度向量的一个坐标并累加，最后归一化；
语义相近（字面重叠多）的文本向量距离就近。
"""

from __future__ import annotations

import hashlib
import os
import re

from langchain_core.embeddings import Embeddings

from backend.config import (
    AI_REQUEST_TIMEOUT,
    EMBED_API_KEY,
    EMBED_API_MODEL,
    EMBED_API_URL,
    TEXT_API_KEY,
    TEXT_API_MODEL,
    TEXT_API_URL,
)


class CharEmbeddings(Embeddings):
    """字符 n-gram 哈希嵌入：零依赖、零联网的教学用嵌入实现。"""

    def __init__(self, dim: int = 256) -> None:
        # 向量维度
        self.dim = dim

    def _hash1(self, text: str) -> int:
        """把一个字符串哈希成稳定整数（md5 前 8 位 hex）。"""
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)

    def _vectorize(self, text: str) -> list[float]:
        """文本 → 归一化向量。"""
        # 去掉所有空白，中英文统一处理
        pure = re.sub(r"\s+", "", text)
        vec = [0.0] * self.dim
        for i in range(len(pure)):
            # 生成 1/2/3 字 shingle（n-gram）
            for n in (1, 2, 3):
                if i + n <= len(pure):
                    shingle = pure[i : i + n]
                    h = self._hash1(shingle)
                    idx = h % self.dim
                    # 用哈希另一段位决定正负号，降低坐标冲突影响
                    sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                    vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入（文档入库用）。"""
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条嵌入（用户提问用）。"""
        return self._vectorize(text)


def has_real_chat() -> bool:
    """是否配置了可用的对话大模型（地址、模型名、密钥三者齐全）。"""
    return bool(TEXT_API_URL and TEXT_API_MODEL and TEXT_API_KEY)


def get_chat_model():
    """创建 LangChain 聊天模型；未配置 Key 时返回 None（教学模拟模式）。"""
    if not has_real_chat():
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": TEXT_API_MODEL,
            "api_key": TEXT_API_KEY,
            "base_url": TEXT_API_URL.rstrip("/"),
            "temperature": 0.3,
            "timeout": AI_REQUEST_TIMEOUT,
        }
        # DeepSeek V4 系列（deepseek-v4-pro/flash）默认开启思维链（thinking）。
        # Agent 工具调用循环默认关闭思维链：响应更快、tool_calls 输出更稳定；
        # 如需更强推理可在 .env 设置 TINYLLM_TEXT_API_THINKING=on。
        if "deepseek.com" in TEXT_API_URL and os.getenv(
            "TINYLLM_TEXT_API_THINKING", "off"
        ).strip().lower() != "on":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        return ChatOpenAI(**kwargs)
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        print(f"[AI工厂] 聊天模型初始化失败，回退教学模式：{exc}")
        return None


def get_embeddings() -> Embeddings:
    """创建嵌入模型：优先真嵌入，未配置/失败时回退教学嵌入。"""
    if EMBED_API_URL and EMBED_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=EMBED_API_MODEL or "text-embedding-3-small",
                api_key=EMBED_API_KEY,
                base_url=EMBED_API_URL.rstrip("/"),
            )
        except Exception as exc:
            print(f"[AI工厂] 嵌入模型初始化失败，回退教学嵌入：{exc}")
    return CharEmbeddings()


def embedding_tag(embeddings: Embeddings) -> str:
    """返回嵌入类型标记，用于向量索引 meta（嵌入算法变了需要重建索引）。"""
    return "char" if isinstance(embeddings, CharEmbeddings) else "openai"
