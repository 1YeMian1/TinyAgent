"""智能助手接口：会话管理 + Agent 对话（LangChain + RAG）。"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth_util import current_user
from backend.database import get_db
from backend.models import ChatMessage, ChatSession, User, utc_now
from backend.schemas import AssistantChatRequest, SessionCreateRequest
from backend.services.assistant import load_history, run_agent

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


def _get_owned_session(db: Session, session_id: int, user_id: int) -> ChatSession:
    """读取会话并校验属主；不存在或不属于当前用户统一返回 404（不泄露存在性）。"""
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


def _parse_json_field(raw: str) -> Any:
    """安全解析消息行中的 JSON 字段（tool_steps / references）。"""
    if not raw:
        return [] if raw == "" else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _message_to_dict(message: ChatMessage) -> dict:
    """消息 ORM 对象 → 前端可用的字典。"""
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "tool_steps": _parse_json_field(message.tool_steps),
        "references": _parse_json_field(message.references),
        "mode": message.mode,
        "created_at": message.created_at,
    }


@router.post("/sessions")
def create_session(
    payload: SessionCreateRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """创建新的助手会话。"""
    title = payload.title.strip() or "新会话"
    session = ChatSession(user_id=user.id, title=title[:120])
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


@router.get("/sessions")
def list_sessions(
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """列出当前用户的全部会话（按最近活跃倒序）。"""
    sessions = db.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
    ).all()
    return [
        {
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": len(session.messages),
        }
        for session in sessions
    ]


@router.get("/sessions/{session_id}")
def get_session(
    session_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """获取会话详情与全部消息。"""
    session = _get_owned_session(db, session_id, user.id)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": [_message_to_dict(message) for message in session.messages],
    }


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """删除会话（级联删除消息）。"""
    session = _get_owned_session(db, session_id, user.id)
    db.delete(session)
    db.commit()
    return {"ok": True}


@router.post("/sessions/{session_id}/chat")
def chat(
    session_id: int,
    payload: AssistantChatRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """向助手发送一条消息，返回 Agent 回答、工具轨迹与引用来源。"""
    session = _get_owned_session(db, session_id, user.id)
    question = payload.message.strip()

    # 首条消息前的历史（用于多轮记忆注入）
    history = load_history(db, session.id)

    # 上一轮助手引用过的故事（教学模式多轮指代用）
    prior_refs: Optional[list[dict]] = None
    last_assistant = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id, ChatMessage.role == "assistant")
        .order_by(ChatMessage.id.desc())
        .limit(1)
    ).first()
    if last_assistant and last_assistant.references:
        parsed = _parse_json_field(last_assistant.references)
        if isinstance(parsed, list) and parsed:
            prior_refs = parsed

    # 运行 Agent（真模型 / 教学模拟双模式）
    result = run_agent(db, question, history=history, prior_refs=prior_refs)

    # 持久化用户消息与助手消息
    is_first_turn = len(session.messages) == 0
    db.add(ChatMessage(session_id=session.id, role="user", content=question, mode=result["mode"]))
    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        tool_steps=json.dumps(result["tool_steps"], ensure_ascii=False),
        references=json.dumps(result["references"], ensure_ascii=False),
        mode=result["mode"],
    )
    db.add(assistant_message)

    # 首条消息用问题前 20 字作为会话标题；每次对话刷新活跃时间
    if is_first_turn:
        session.title = question[:20]
    session.updated_at = utc_now()

    db.commit()
    db.refresh(assistant_message)
    return {
        "answer": result["answer"],
        "tool_steps": result["tool_steps"],
        "references": result["references"],
        "mode": result["mode"],
        "message_id": assistant_message.id,
    }
