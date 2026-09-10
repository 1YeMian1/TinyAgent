"""
TinyLLM-Story 应用配置

配置优先级：
1. 系统环境变量
2. 项目根目录 .env 文件
3. 本地默认配置
"""

# 开启类型注解延迟解析
from __future__ import annotations

# 导入系统环境变量模块
import os

# 导入路径处理模块
from pathlib import Path


# 加载.env配置文件
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# 获取项目根目录
ROOT = Path(__file__).resolve().parents[1]


# 加载项目根目录.env文件
if load_dotenv:

    # 设置可能存在的.env路径
    for env_file in (
        ROOT / ".env",
        Path.cwd() / ".env"
    ):

        # 判断文件是否存在
        if env_file.exists():

            # 加载环境变量，不覆盖系统变量
            load_dotenv(
                env_file,
                override=False,
                encoding="utf-8"
            )

            # 找到配置后停止搜索
            break



# ==============================
# 环境变量读取函数
# ==============================


def _env_str(
    name: str,
    *aliases: str,
    default: str = ""
) -> str:
    """
    获取字符串类型环境变量。

    参数：
        name:
            主环境变量名称。

        aliases:
            备用环境变量名称。

        default:
            默认返回值。

    返回：
        str:
            环境变量字符串。
    """

    # 遍历变量名称
    for key in (name, *aliases):

        # 获取环境变量
        value = os.getenv(key)

        # 判断变量是否有效
        if value:

            # 删除首尾空格
            value = value.strip()

            # 删除包裹引号
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]

            # 返回变量值
            return value

    # 返回默认值
    return default



def _env_int(
    name: str,
    default: int
) -> int:
    """
    获取整数类型环境变量。

    参数：
        name:
            环境变量名称。

        default:
            默认整数值。

    返回：
        int:
            转换后的整数。
    """

    try:

        # 获取并转换整数
        return int(
            os.getenv(
                name,
                default
            )
        )

    except ValueError:

        # 转换失败返回默认值
        return default



def _env_float(
    name: str,
    default: float
) -> float:
    """
    获取浮点类型环境变量。

    参数：
        name:
            环境变量名称。

        default:
            默认浮点值。

    返回：
        float:
            转换后的浮点数。
    """

    try:

        # 获取并转换浮点数
        return float(
            os.getenv(
                name,
                default
            )
        )

    except ValueError:

        # 转换失败返回默认值
        return default



# ==============================
# 文件路径配置
# ==============================


# 前端静态文件目录
STATIC_DIR = ROOT / "fronted"


# 故事数据集路径
DATASET_PATH = (
    ROOT /
    "src" /
    "data" /
    "stories_dataset_v2.json"
)


# 图片生成保存目录
GENERATED_IMAGE_DIR = (
    STATIC_DIR /
    "generated"
)



# ==============================
# 数据库配置
# ==============================


# 获取数据库连接地址
DATABASE_URL: str = _env_str(
    "TINYLLM_DATABASE_URL",
    default=f"sqlite:///{ROOT / 'tinyllm.db'}"
)



# ==============================
# JWT安全配置
# ==============================


# 获取JWT密钥
JWT_SECRET: str = _env_str(
    "TINYLLM_JWT_SECRET",
    default="tinyllm-development-secret-change-me"
)


# 设置JWT加密算法
JWT_ALGORITHM = "HS256"


# 获取Token有效时间
TOKEN_TTL_HOURS: int = _env_int(
    "TINYLLM_TOKEN_TTL_HOURS",
    24
)



# ==============================
# AI文本生成配置
# ==============================


# 获取本地模型路径
MODEL_PATH: str = _env_str(
    "TINYLLM_MODEL_PATH"
)


# 获取文本生成API地址
TEXT_API_URL: str = _env_str(
    "TINYLLM_TEXT_API_URL",
    "OPENAI_BASE_URL"
)


# 获取文本生成API密钥
TEXT_API_KEY: str = _env_str(
    "TINYLLM_TEXT_API_KEY",
    "OPENAI_API_KEY"
)


# 获取文本模型名称
TEXT_API_MODEL: str = _env_str(
    "TINYLLM_TEXT_API_MODEL",
    "OPENAI_MODEL"
)



# ==============================
# AI图片生成配置
# ==============================


# 获取图片API地址
IMAGE_API_URL: str = _env_str(
    "TINYLLM_IMAGE_API_URL"
)


# 获取图片API密钥
IMAGE_API_KEY: str = _env_str(
    "TINYLLM_IMAGE_API_KEY",
    "OPENAI_API_KEY"
)


# 获取图片模型名称
IMAGE_API_MODEL: str = _env_str(
    "TINYLLM_IMAGE_API_MODEL"
)


# 获取图片尺寸
IMAGE_SIZE: str = _env_str(
    "TINYLLM_IMAGE_SIZE",
    default="1024x1024"
)



# ==============================
# AI请求配置
# ==============================


# 获取请求超时时间
AI_REQUEST_TIMEOUT: float = _env_float(
    "TINYLLM_AI_REQUEST_TIMEOUT",
    120.0
)



# ==============================
# 智能助手配置（LangChain + RAG + Agent）
# ==============================


# FAISS 向量索引落盘目录（故事库 RAG）
ASSISTANT_INDEX_DIR = ROOT / "data" / "story_faiss_index"


# RAG 文本切块参数（中文字符数）
RAG_CHUNK_SIZE: int = _env_int(
    "TINYLLM_RAG_CHUNK_SIZE",
    400
)


RAG_CHUNK_OVERLAP: int = _env_int(
    "TINYLLM_RAG_CHUNK_OVERLAP",
    80
)


# 每次语义检索取回的片段数
RAG_TOP_K: int = _env_int(
    "TINYLLM_RAG_TOP_K",
    3
)


# 多轮对话注入模型的最近消息条数
ASSISTANT_HISTORY_LIMIT: int = _env_int(
    "TINYLLM_ASSISTANT_HISTORY_LIMIT",
    10
)


# Agent 单次任务最大工具调用轮数（安全阀，防止死循环烧 token）
ASSISTANT_MAX_ROUNDS: int = _env_int(
    "TINYLLM_ASSISTANT_MAX_ROUNDS",
    5
)


# 嵌入模型独立配置（DeepSeek 等纯对话厂商不支持 embedding，需单独配置）
# 留空时使用内置"教学嵌入"（字符哈希向量），离线可运行
EMBED_API_URL: str = _env_str(
    "TINYLLM_EMBED_API_URL",
    "OPENAI_BASE_URL"
)


EMBED_API_KEY: str = _env_str(
    "TINYLLM_EMBED_API_KEY",
    "OPENAI_API_KEY"
)


EMBED_API_MODEL: str = _env_str(
    "TINYLLM_EMBED_API_MODEL",
    default="text-embedding-3-small"
) if _env_str("TINYLLM_EMBED_API_URL", "OPENAI_BASE_URL") else ""



# ==============================
# 跨域配置
# ==============================


# 获取允许访问来源
_cors_raw = _env_str(
    "TINYLLM_CORS_ORIGINS",
    default="*"
)


# 转换跨域字符串为列表
CORS_ORIGINS = [
    origin.strip()
    for origin in _cors_raw.split(",")
    if origin.strip()
]


# 防止空列表
if not CORS_ORIGINS:
    CORS_ORIGINS = ["*"]



# ==============================
# 故事分类配置
# ==============================


# 定义故事分类
CATEGORIES = [
    {
        "key": "",
        "label": "全部"
    },
    {
        "key": "动物故事",
        "label": "🐱 动物故事"
    },
    {
        "key": "童话故事",
        "label": "🏰 童话故事"
    },
    {
        "key": "成长故事",
        "label": "🌱 成长故事"
    },
    {
        "key": "冒险故事",
        "label": "🗺️ 冒险故事"
    },
    {
        "key": "温馨故事",
        "label": "🌙 温馨故事"
    },
    {
        "key": "友谊故事",
        "label": "🤝 友谊故事"
    },
    {
        "key": "想象故事",
        "label": "✨ 想象故事"
    }
]


# 根据分类生成emoji映射
EMOJIS = {
    item["key"]: item["label"].split(" ", 1)[0]
    for item in CATEGORIES
    if item["key"]
}