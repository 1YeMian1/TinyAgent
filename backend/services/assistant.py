"""智能助手 Agent 服务：LangChain 工具定义 + 手写 ReAct 循环（双模式）。

- 真模型模式（配置了 TINYLLM_TEXT_API_*）：ChatOpenAI.bind_tools → AIMessage.tool_calls
  → 执行工具 → ToolMessage 回传 → 再次调用模型，直到模型给出最终回答；
- 教学模拟模式（无 Key）：用规则决策模拟同一个"思考→行动→观察"循环，
  工具全部真实执行（检索/SQL/创作），最终回答为资料回显，全流程离线可跑。

两种模式共用同一套工具、同一个安全阀（ASSISTANT_MAX_ROUNDS）和同一份轨迹记录。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import ASSISTANT_HISTORY_LIMIT, ASSISTANT_MAX_ROUNDS, EMOJIS
from backend.models import ChatMessage, Story
from backend.schemas import GenerateRequest
from backend.services import rag_store
from backend.services.ai_factory import get_chat_model
from backend.services.generator_v2 import generate_content, normalize_category


# 系统提示词：告诉模型它有哪些工具、什么时候用、回答边界
SYSTEM_PROMPT = (
    "你是 TinyStory 儿童故事平台的智能助手，只能使用中文回答。\n"
    "你可以使用以下工具：\n"
    "1. retrieve_story_knowledge：语义检索故事库片段——回答关于故事情节、角色、寓意的"
    "问题前必须先调用；\n"
    "2. search_stories：按关键词搜索故事标题与正文；\n"
    "3. get_story：按故事 ID 读取完整故事正文；\n"
    "4. count_story_words：统计指定故事的正文字数；\n"
    "5. create_story：根据主题创作一篇新儿童故事，创作结果会保存到故事库。\n"
    "规则：平台故事相关的问题必须依据检索到的资料回答，资料没有就如实说明，不要编造；"
    "工具返回的内容是给你看的观察结果，不要把工具名或参数直接抛给用户；"
    "创作成功后告诉用户故事标题。回答简洁、友好，适合儿童故事平台。"
)


def build_tools(db: Session, refs: list[dict], steps: list[dict]) -> list:
    """构造本次对话可用的工具集合。

    refs/steps 为可变列表，工具执行时向其中追加引用来源与调用轨迹。
    """

    def add_ref(story_id: Optional[int], title: str) -> None:
        """记录引用来源（按 story_id 去重）。"""
        if not story_id:
            return
        if not any(item.get("story_id") == story_id for item in refs):
            refs.append({"story_id": int(story_id), "title": title})

    @tool
    def retrieve_story_knowledge(query: str) -> str:
        """语义检索故事库中与问题相关的故事片段。

        当用户询问故事情节、角色、寓意，或提到某类故事时使用。
        Args:
            query: 用户问题或检索关键词，例如"小鸭子学游泳的故事"。
        Returns:
            相关故事片段（带编号与标题）；没有检索到时返回提示。
        """
        docs = rag_store.retrieve(query)
        if not docs:
            return "没有检索到与该问题相关的故事片段。"
        blocks = []
        for i, doc in enumerate(docs, start=1):
            add_ref(doc["story_id"], doc["title"])
            blocks.append(f"[{i}] 出自《{doc['title']}》（分类：{doc['category']}）\n{doc['content']}")
        return "\n\n".join(blocks)

    @tool
    def search_stories(keyword: str) -> str:
        """按关键词在故事库中搜索故事标题与正文。

        当用户想找某类/某个主题的故事清单时使用。
        Args:
            keyword: 中文关键词，例如"小熊"或"勇气"。
        Returns:
            匹配故事的 JSON 数组（id/title/category），最多 5 条。
        """
        like = f"%{keyword.strip()}%"
        rows = db.scalars(
            select(Story)
            .where(Story.title.like(like) | Story.content.like(like))
            .order_by(Story.id)
            .limit(5)
        ).all()
        for row in rows:
            add_ref(row.id, row.title)
        if not rows:
            return f"没有找到包含“{keyword}”的故事。"
        return json.dumps(
            [{"id": row.id, "title": row.title, "category": row.category} for row in rows],
            ensure_ascii=False,
        )

    @tool
    def get_story(story_id: int) -> str:
        """按故事 ID 获取一篇故事的完整正文。

        当已经知道故事 ID、需要读取全文时使用。
        Args:
            story_id: 故事的数字 ID。
        Returns:
            故事标题与正文；找不到时返回提示。
        """
        story = db.get(Story, int(story_id))
        if not story:
            return f"未找到 ID 为 {story_id} 的故事。"
        add_ref(story.id, story.title)
        return f"《{story.title}》（分类：{story.category}）\n{story.content}"

    @tool
    def count_story_words(story_id: int) -> str:
        """统计指定故事的正文字数。

        当用户问某篇故事"有多少字/多长"时使用。
        Args:
            story_id: 故事的数字 ID。
        Returns:
            包含字数统计结果的文本。
        """
        story = db.get(Story, int(story_id))
        if not story:
            return f"未找到 ID 为 {story_id} 的故事，无法统计字数。"
        add_ref(story.id, story.title)
        return f"《{story.title}》正文共 {story.words} 个字符。"

    @tool
    def create_story(theme: str, category: str = "想象故事", character: str = "小主角") -> str:
        """根据主题创作一篇新的儿童故事并保存到故事库。

        当用户要求"写/创作/编一个故事"时使用。
        Args:
            theme: 故事主题，例如"分享""勇气"。
            category: 故事分类，例如 动物故事/成长故事；不确定时给"想象故事"。
            character: 主角名称，例如"小狐狸"。
        Returns:
            新故事的 JSON（id/title/preview）。
        """
        category = normalize_category(category, theme)
        payload = GenerateRequest(
            theme=theme[:120],
            category=category,
            character=character[:80] or "小主角",
            length="short",
        )
        content, provider = generate_content(payload)
        story = Story(
            title=f"{theme.strip()[:20]}：{payload.character}的故事",
            category=category,
            summary=content[:100],
            content=content,
            words=len(content),
            emoji=EMOJIS.get(category, "✨"),
            cover_type="c7",
        )
        db.add(story)
        db.flush()
        # 增量加入 RAG 向量索引（失败不影响创作结果）
        rag_store.add_story(story)
        db.commit()
        db.refresh(story)
        add_ref(story.id, story.title)
        return json.dumps(
            {"id": story.id, "title": story.title, "preview": content[:80], "provider": provider},
            ensure_ascii=False,
        )

    return [retrieve_story_knowledge, search_stories, get_story, count_story_words, create_story]


def load_history(db: Session, session_id: int, limit: int = ASSISTANT_HISTORY_LIMIT) -> list:
    """读取该会话最近 N 条消息，转为 LangChain 消息对象（多轮记忆注入）。"""
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    ).all()
    rows.reverse()
    messages = []
    for row in rows:
        cls = HumanMessage if row.role == "user" else AIMessage
        messages.append(cls(content=row.content))
    return messages


def _record_step(steps: list[dict], name: str, args: Any, result: str) -> None:
    """记录一条工具调用轨迹（结果只保留预览，避免消息体过大）。"""
    preview = str(result).replace("\n", " ")
    steps.append(
        {
            "tool": name,
            "args": args if isinstance(args, dict) else {"value": str(args)},
            "result": preview[:150],
        }
    )


def _run_real(
    db: Session, question: str, history: list, tools: list, refs: list, steps: list
) -> str:
    """真模型模式：bind_tools + tool_calls + ToolMessage 的手写 ReAct 循环。"""
    llm = get_chat_model()
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=question)]

    answer = ""
    for _ in range(ASSISTANT_MAX_ROUNDS):
        ai_message: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        # 模型不再请求工具 → 本轮为最终回答
        if not getattr(ai_message, "tool_calls", None):
            answer = str(ai_message.content or "").strip()
            break

        # 执行模型请求的每一个工具，并把观察结果作为 ToolMessage 回传
        for call in ai_message.tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool_obj = tools_by_name.get(name)
            try:
                result = tool_obj.invoke(args) if tool_obj else f"未知工具：{name}"
            except Exception as exc:
                result = f"工具 {name} 执行出错：{exc}"
            _record_step(steps, name, args, str(result))
            messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", "")))
    else:
        # 安全阀：达到最大轮数仍未给最终回答
        answer = "我围绕你的问题查询了多轮资料，暂时没有形成完整答案。请换个问法再试一次。"
    return answer


def _run_simulation(
    db: Session, question: str, tools: list, refs: list, steps: list
) -> str:
    """教学模拟模式：用规则决策演示同一个 ReAct 循环（工具全部真实执行）。"""
    tools_by_name = {t.name: t for t in tools}

    def call(name: str, **kwargs: Any) -> str:
        result = tools_by_name[name].invoke(kwargs)
        _record_step(steps, name, kwargs, str(result))
        return str(result)

    # 意图 1：创作故事
    if re.search(r"写|创作|编|生成", question) and re.search(r"故事", question):
        # 从问句中抽取主题：优先"关于X的"，其次"写(一个/一篇)X故事"
        theme = question.strip(" ？?。！!")
        theme_match = re.search(r"关于(.+?)的", question) or re.search(
            r"写(?:一个|一篇)?(.{1,20}?)故事", question
        )
        if theme_match:
            theme = theme_match.group(1).strip(" ，,。！!？?")
        character_match = re.search(r"小[\u4e00-\u9fa5]{1,2}", question)
        result = call(
            "create_story",
            theme=theme[:60],
            character=character_match.group(0) if character_match else "小主角",
        )
        data = json.loads(result)
        return (
            f"（教学模拟模式：未配置大模型 API Key，以下为工具执行结果）\n\n"
            f"已调用 create_story 工具为你创作新故事《{data['title']}》（故事 ID：{data['id']}），"
            f"新故事已保存到故事库并加入语义索引，可在故事库中查看。\n"
            f"故事开头：{data['preview']}……\n\n"
            f"配置 TINYLLM_TEXT_API_KEY 后，助手将由大模型自主规划并直接生成完整故事。"
        )

    # 意图 2：统计字数（需要先确定目标故事）
    if re.search(r"多少字|字数|几个字|多长", question):
        if refs:
            result = call("count_story_words", story_id=refs[-1]["story_id"])
        else:
            call("retrieve_story_knowledge", query=question)
            if refs:
                result = call("count_story_words", story_id=refs[-1]["story_id"])
            else:
                return "（教学模拟模式）没有找到相关故事，无法统计字数。请先提到某篇故事。"
        return f"（教学模拟模式）已调用字数统计工具：{result}"

    # 意图 3：默认 → RAG 语义检索，资料回显
    result = call("retrieve_story_knowledge", query=question)
    if refs:
        titles = "、".join(f"《{item['title']}》" for item in refs)
        return (
            "（教学模拟模式：未配置大模型 API Key，以下为故事库中检索到的资料）\n\n"
            f"{result}\n\n"
            f"相关故事：{titles}。可点击下方“引用来源”查看完整故事。\n"
            "配置 TINYLLM_TEXT_API_KEY 后，助手将基于这些资料自动组织回答。"
        )
    return f"（教学模拟模式）{result}"


def run_agent(
    db: Session,
    question: str,
    history: Optional[list] = None,
    prior_refs: Optional[list[dict]] = None,
) -> dict:
    """运行一次 Agent 对话。

    prior_refs：上一轮助手回答引用过的故事（教学模式多轮指代用）。
    返回 {answer, tool_steps, references, mode}。
    """
    refs: list[dict] = list(prior_refs or [])
    steps: list[dict] = []
    tools = build_tools(db, refs, steps)
    history = history or []

    llm = get_chat_model()
    if llm is not None:
        try:
            answer = _run_real(db, question, history, tools, refs, steps)
            mode = "real"
        except Exception as exc:
            # 真模型不可用（密钥无效/网络错误/限流等）时，自动降级到教学模拟模式，
            # 避免接口 500；同时在回答开头说明降级原因。
            print(f"[助手] 真模型调用失败，降级到教学模拟模式：{exc}")
            sim_answer = _run_simulation(db, question, tools, refs, steps)
            answer = (
                f"（注意：当前配置的大模型调用失败，已自动切换到教学模拟模式。"
                f"失败原因：{exc}\n\n{sim_answer}"
            )
            mode = "simulation"
    else:
        mode = "simulation"
        answer = _run_simulation(db, question, tools, refs, steps)

    return {
        "answer": answer,
        "tool_steps": steps,
        "references": refs,
        "mode": mode,
    }
