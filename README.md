# TinyAgent / TinyLLM-Story

用 AI 为孩子生成、检索、朗读儿童故事的全栈平台项目。前端采用轻量响应式页面，后端基于 FastAPI 搭建。平台融合了故事库管理、用户认证与收藏、AI 故事生成（支持多级降级回退）、智能配图（兼容远程模型与本地 SVG 矢量绘图）、浏览器语音朗读，以及基于 LangChain + FAISS + ReAct Agent 的儿童知识助手。

平台设计了**真模型模式**与**教学模拟模式**双通道架构：未配置任何大模型 API Key 或离线状态下，系统仍可完整自闭环运行（故事走本地启发式模板引擎，插图走本地数学算法生成 SVG 绘图，智能助手走规则驱动的意图分流与真实工具执行）。

---

## 目录
1. [技术栈详解](#技术栈详解)
2. [关键功能实现原理](#关键功能实现原理)
3. [项目目录结构](#项目目录结构)
4. [环境配置与快速启动](#环境配置与快速启动)
5. [主要 API 接口规范](#主要-api-接口规范)
6. [注意事项与说明](#注意事项与说明)

---

## 技术栈详解

### 1. 后端架构
- **Web 框架**：`FastAPI (>=0.100.0)` + `Uvicorn`。采用 ASGI 异步生命周期管理（`lifespan`），在应用启动时自动检测并初始化 SQLite 故事库数据，并挂载前端静态站点。
- **ORM 与数据持久化**：`SQLAlchemy 2.0`。使用强类型 `Mapped` 与 `mapped_column` 声明模型，默认连接本地 SQLite 数据库文件 `tinyllm.db`。
- **参数校验与序列化**：`Pydantic v2`。支持复杂字符串校验、篇幅枚举约束及邮箱格式验证（结合 `email-validator`）。
- **安全认证与权限**：
  - 密码哈希：PBKDF2-HMAC-SHA256，内置 210,000 次高强度迭代与安全随机加盐（16 字节）。
  - 会话机制：`PyJWT` 签发 HS256 Token，配合 FastAPI `Depends(current_user)` 依赖注入实现细粒度接口鉴权与可选匿名访问（`optional_user`）。
- **环境变量管理**：`python-dotenv`。三级分层配置（系统环境变量 > 项目根目录 `.env` > 代码缺省值），保护生产私密配置。

### 2. AI、RAG 与智能代理（Agent）
- **LangChain 核心生态**：`langchain-core`、`langchain-openai`、`langchain-community`、`langchain-text-splitters`。
- **向量检索与知识库（RAG）**：
  - 文本切分：`RecursiveCharacterTextSplitter`，结合中文标点切块（默认 400 字符，重叠 80 字符）。
  - 向量存储：`FAISS` 本地索引持久化（`data/story_faiss_index/`）。带状态元数据（`meta.json`），数据变动或嵌入算法升级时自动触发全量重建；创作新故事时支持增量实时写入（`add_story`）。
  - 嵌入通道：配置 `TINYLLM_EMBED_*` 时接入 `OpenAIEmbeddings`；未配置时无缝降级至内置 `CharEmbeddings`（256 维字符 n-gram 哈希向量），实现 0 成本、无外网依赖的教学向量语义检索。
- **故事生成引擎（三级降级链路）**：
  1. 第一优先级：OpenAI 兼容对话模型接口（如 DeepSeek、OpenAI、ChatGLM 等）；
  2. 第二优先级：本地 HuggingFace Transformers 因果生成模型；
  3. 第三优先级：内置纯 Python 启发式规则模板故事引擎，具备丰富的情节冲突、伙伴协作与品格教育脉络。
- **ReAct 智能助手**：
  - 基于工具注册（`@tool`）：语义检索 `retrieve_story_knowledge`、关键字模糊检索 `search_stories`、全文读取 `get_story`、字符统计 `count_story_words`、创作并落库 `create_story`。
  - 真模型模式下由大模型自主规划 `tool_calls`，回传 `ToolMessage` 观察结果，并设最大循环安全阀（默认 5 轮）；
  - 离线或异常时自动降级至模拟规划执行，工具均真实运行并记录完整调用轨迹（`tool_steps`）与关联引用（`references`）。
- **智能配图服务**：
  - 远程接口：兼容 OpenAI `/images/generations` 绘图 API，支持 URL 与 Base64 图片下载落盘到 `fronted/generated/story-{id}/`；
  - 本地回退：`image_generator.py` 内置 SVG 算法绘图，根据故事属性和预设调色板动态生成高颜值矢量插图。

### 3. 前端界面
- 原生轻量架构：纯 HTML5 + CSS3 + 现代 JavaScript（ES6+），无打包工具，存放于 `fronted/`。
- 交互特性：响应式布局、Flex/Grid 排版、CSS 动画、主题 Emoji 卡片、Toast 轻提示、防抖搜索、SSE 流式消息消费（`EventSource`）、浏览器原生 Web Speech API 朗读。

---

## 关键功能实现原理

### 1. 应用启动与生命周期管理
`backend/main.py` 使用 `@asynccontextmanager` 定义生命周期：
1. 启动时执行 `seed_database()`：
   - 使用 SQLAlchemy 检查 SQLite 中的 `stories` 数据表。
   - 若表为空，自动读取 `src/data/stories_dataset_v2.json` 故事数据，解析其分类、正文，并根据分类自动映射封面和 Emoji 图标，批量插入数据库。
2. 静态资源挂载：检测 `fronted` 目录存在后，通过 `StaticFiles(directory=STATIC_DIR, html=True)` 挂载到根路径 `/`，使得浏览器访问 `http://localhost:8000` 即可直接展示完整前端。
3. 跨域配置：注册 `CORSMiddleware`，支持来自自定义前端来源或全开放跨域访问。

### 2. 身份认证与权限控制
- 注册接口（`POST /api/v1/auth/register`）：检查邮箱全局唯一性，将明文密码与 16 字节随机 Salt 结合，经过 210,000 次 PBKDF2-SHA256 哈希后存入数据库。
- 登录接口（`POST /api/v1/auth/login`）：采用 `hmac.compare_digest` 安全比对密码摘要，验证成功后签发包含用户 ID 和有效期的 HS256 JWT Token。
- 依赖注入：`auth_util.py` 提供 `current_user`（必须登录）与 `optional_user`（未登录返回 None）两种解析方式，前端将 Token 缓存在 `localStorage` 中并在每次请求时通过 `Authorization: Bearer <token>` 携带。

### 3. AI 故事生成与流式输出
- 多级降级策略：在 `backend/services/generator_v2.py` 中，生成流程优先调用 OpenAI 兼容接口，若未配置或异常则尝试本地模型，最后自动降级至本地故事引擎，确保任何环境下生成接口均 100% 可用。
- 字符与主题约束：系统针对 `short`（约 180~320 字）、`medium`（约 420~650 字）、`long`（约 750~1100 字）定制系统提示词，严守安全、温馨、无说教、适合 3~10 岁儿童的故事准则。
- SSE 流式接口（`GET /api/v1/generate/story/stream`）：基于 FastAPI `StreamingResponse`，按 24 字符切片以 `text/event-stream` 格式持续推送，以友好动效实时呈现生成过程。

### 4. 绘图与语音朗读
- 智能配图（`backend/services/image_generator.py`）：分析故事开端、转折与结尾，生成绘图 Prompt。远程调用失败时，本地采用动态 SVG 算法，结合故事 ID 散列色彩与儿童手绘元素即时渲染高品质矢量卡片。
- 语音朗读：前端 `story-detail.html` 采用浏览器原生 Web Speech API，提取故事正文，支持断句朗读、播放/暂停控制与实时进度模拟。

### 5. RAG 故事知识库与智能助手
- 向量存储与增量同步（`backend/services/rag_store.py`）：
  - 启动或初次检索时，自动将所有故事分割并计算向量存入 FAISS，并在磁盘生成 `meta.json` 记录故事版本信息；
  - 助手通过工具创作出新故事后，自动调用 `add_story()` 将新故事增量加入向量数据库，保证知识库实时更新。
- 双模式 ReAct Agent（`backend/services/assistant.py`）：
  - 绑定 5 个核心工具函数：`retrieve_story_knowledge`、`search_stories`、`get_story`、`count_story_words`、`create_story`。
  - **真模型模式**：通过 LangChain 的 `bind_tools` 机制循环解析模型 tool_calls，调用工具后回传观察，完成多轮推理后给出回答；
  - **教学模拟模式**：在无 API Key 场景下，内置意图识别逻辑，精准调用相应工具完成检索、统计或创作，并返回清晰的过程展示，保障离线教学与演示体验。


---

## 项目目录结构

```text
Tiny_Agent/
├── backend/                      # 后端核心服务
│   ├── main.py                   # FastAPI 应用入口与生命周期管理
│   ├── config.py                 # 环境变量解析与路径配置
│   ├── database.py               # 数据库引擎、SessionLocal 与数据初始化
│   ├── models.py                 # SQLAlchemy 数据表模型 (User, Story, History 等)
│   ├── schemas.py                # Pydantic 请求与响应格式定义
│   ├── auth_util.py              # 密码加盐哈希与 JWT 校验逻辑
│   ├── serializers.py            # 故事对象序列化与格式化
│   ├── routers/                  # 接口路由层
│   │   ├── system_router.py      # 分类等基础信息接口
│   │   ├── auth_router.py        # 用户注册、登录、个人中心
│   │   ├── storeis_router.py     # 故事列表、详情、搜索、收藏与足迹
│   │   ├── generate.py           # 故事生成、SSE 流式推送、绘图
│   │   └── assistant.py          # 智能助手会话管理与 Agent 问答接口
│   └── services/                 # 核心业务与 AI 服务层
│       ├── ai_factory.py         # 聊天模型与嵌入模型工厂（含教学哈希嵌入）
│       ├── generator_v2.py       # 故事生成三级回退引擎
│       ├── image_generator.py    # 绘图模型接口与本地 SVG 矢量图渲染器
│       ├── rag_store.py          # FAISS 向量索引与全量/增量构建
│       └── assistant.py          # ReAct Agent 手写循环与多工具编排
├── fronted/                      # 前端静态站点 (纯原生 Web 技术)
│   ├── index.html                # 故事库首页（分类筛选、搜索、卡片流）
│   ├── story-generate.html       # AI 故事创作工作台
│   ├── story-detail.html         # 故事阅读器、绘图展示与语音朗读
│   ├── assistant.html            # AI 智能助手会话页面
│   ├── login.html / register.html# 登录与注册
│   ├── profile.html              # 个人中心（收藏夹与阅读历史）
│   ├── css/style.css             # 统一童趣设计规范样式表
│   └── js/
│       ├── common.js             # Storage 存储、Auth 状态、Toast、日期格式化
│       └── api.js                # API 统一调用封装层
├── Langchain/                    # LangChain 教学与概念验证模块
│   ├── config.py                 # 独立配置加载
│   ├── llm_factory.py            # 模型工厂示例
│   ├── Prompt_llm.py             # Runnable 与 RunnableLambda 链式调用示例
│   ├── Prompt_RAFT.py            # RAFT 结构化角色提示词示例
│   └── memory_llm.py             # ConversationBufferMemory 多轮记忆演示
├── src/data/                     # 故事语料与预置数据
│   ├── stories_dataset_v2.json   # 启动自动导入的故事集
│   └── gushi365_data/            # 完整儿童故事库
├── test/                         # 模型试验与转换脚本 (ViT / DETR / DistilBERT 等)
├── .gitignore                    # Git 版本控制忽略配置
├── requirements.txt              # 项目全量依赖清单
└── README.md                     # 项目技术说明文档
```

---

## 环境配置与快速启动

### 1. 基础环境
- 操作系统：Windows / Linux / macOS
- Python 版本：3.10 及以上（推荐 Python 3.11）

### 2. 安装步骤
在项目根目录下打开终端，创建虚拟环境并安装依赖：
```bash
# 创建并激活虚拟环境
python -m venv .venv

# Windows 环境激活:
.venv\Scripts\activate
# Linux / macOS 环境激活:
# source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量
复制根目录的 `.env.example`（或新建 `.env`），填入大模型配置（若不配置，系统将自动进入本地/教学模式）：
```env
# 对话模型（兼容 OpenAI 规范，支持 DeepSeek 等平台）
TINYLLM_TEXT_API_URL=https://api.deepseek.com
TINYLLM_TEXT_API_KEY=sk-your-api-key
TINYLLM_TEXT_API_MODEL=deepseek-chat

# 嵌入模型（可选；不填则使用内置 CharEmbeddings 字符哈希嵌入）
TINYLLM_EMBED_API_URL=https://api.openai.com/v1
TINYLLM_EMBED_API_KEY=sk-your-openai-key
TINYLLM_EMBED_API_MODEL=text-embedding-3-small

# 图像生成模型（可选）
TINYLLM_IMAGE_API_URL=https://api.openai.com/v1
TINYLLM_IMAGE_API_KEY=sk-your-openai-key
TINYLLM_IMAGE_API_MODEL=dall-e-3
```

### 4. 运行应用
在**项目根目录**运行启动命令：
```bash
uvicorn backend.main:app --reload --port 8000
```
- 前端首页：访问 `http://localhost:8000`
- 交互式 Swagger API 文档：访问 `http://localhost:8000/docs`
- 首次运行会自动创建 SQLite 数据库 `tinyllm.db` 并写入预置故事库。

---

## 主要 API 接口规范

| 分类 | 请求方法 | 路径 | 功能说明 |
|:---|:---|:---|:---|
| **系统** | `GET` | `/api/v1/categories` | 获取系统故事分类列表、全量 Emoji 与说明元数据 |
| **认证** | `POST` | `/api/v1/auth/register` | 用户注册，返回 Token 与个人信息 |
| **认证** | `POST` | `/api/v1/auth/login` | 用户登录并获取 JWT Token |
| **认证** | `GET` | `/api/v1/auth/profile` | 获取当前用户信息及收藏/历史统计 |
| **故事** | `GET` | `/api/v1/stories` | 分页、分类过滤与关键字搜索故事列表 |
| **故事** | `GET` | `/api/v1/stories/{id}` | 获取单个故事详情及插图 |
| **故事** | `POST` | `/api/v1/stories/{id}/favorite` | 切换故事收藏状态 |
| **故事** | `GET` | `/api/v1/stories/favorites` | 获取我的收藏列表 |
| **故事** | `GET` | `/api/v1/stories/history` | 获取浏览历史记录 |
| **生成** | `POST` | `/api/v1/generate/story` | AI 创作故事并自动入库与生成插图 |
| **生成** | `GET` | `/api/v1/generate/story/stream` | SSE 流式生成故事文本推送 |
| **生成** | `POST` | `/api/v1/generate/images` | 为已有故事重新生成绘图 |
| **助手** | `POST` | `/api/v1/assistant/sessions` | 创建新的助手对话会话 |
| **助手** | `GET` | `/api/v1/assistant/sessions` | 获取当前用户的全部对话列表 |
| **助手** | `GET` | `/api/v1/assistant/sessions/{id}` | 获取指定会话的历史消息及工具轨迹 |
| **助手** | `POST` | `/api/v1/assistant/sessions/{id}/chat` | 发送问题，执行 ReAct Agent 问答与工具调用 |

---

## 注意事项与说明
1. **静态目录命名**：前端资源目录为 `fronted`，与 `backend/config.py` 中的 `STATIC_DIR` 保持一致。
2. **离线高可用**：即使断网或不提供任何外部大模型 API Key，所有核心功能均有完备的本地算法或教学实现托底。
3. **敏感信息保护**：项目包含的 `.gitignore` 已严格排除本地数据库（`*.db`）、向量索引（`data/story_faiss_index/`）、环境文件（`.env`）以及运行期生成的配图目录。
