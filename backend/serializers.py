#当前模块提供接口共用的响应数据序列化方法
from __future__ import annotations
from typing import Any
#故事模型
from backend.models import Story


# 系统预设儿童故事朗读角色列表（多元音色）
PRESET_VOICES: list[dict[str, str]] = [
    {
        "id": "browser",
        "name": "标准旁白",
        "avatar": "🎤",
        "description": "沉稳亲切，适合睡前故事与通篇朗读",
        "gender": "neutral",
    },
    {
        "id": "sister_gentle",
        "name": "温柔姐姐",
        "avatar": "🌸",
        "description": "音色清亮柔和，适合温馨与成长故事",
        "gender": "female",
    },
    {
        "id": "grandpa_tree",
        "name": "博学树爷爷",
        "avatar": "🌳",
        "description": "磁性深厚语速舒缓，适合寓言与童话冒险",
        "gender": "male",
    },
    {
        "id": "rabbit_cute",
        "name": "活泼小萌兔",
        "avatar": "🐰",
        "description": "灵动欢快音调微扬，适合动物故事与欢快场景",
        "gender": "child",
    },
]


#定义语音信息查询函数
def voices() -> dict[str, list[dict[str, str]]]:
    """返回当前系统支持的角色音色列表元数据"""
    return {
        "voices": PRESET_VOICES
    }


#定义函数进行故事数据格式转换
def story_summary(story: Story, include_content: bool = False) -> dict[str, Any]:
    #创建故事基础信息字典，用于返回概要信息
    item = {
        "id": story.id,
        "title": story.title,
        "category": story.category,
        "summary": story.summary,
        "words": story.words,
        "emoji": story.emoji,
        "coverType": story.cover_type,
        "date": story.created_at.isoformat(),
        "voices": [v["name"] for v in PRESET_VOICES]
    }

    #判断是否需要返回完整的故事内容
    if include_content:
        item.update(
            {
                "content": story.content,
                "images": [image.url for image in story.images],
                "voices": voices()["voices"]
            }
        )

    return item
