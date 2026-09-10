#当前模块提供接口共用的响应数据序列化方法
from  __future__ import annotations
from typing import  Any
#故事模型
from backend.models import Story


#定义语音信息查询函数
def voices()->dict[str,list[dict[str,str]]]:
    return {
        "voices":[
            {
                "id":"browser",
                "name":"标准旁白",
                "avatar":"🎤",
                "description":"使用浏览器语言合成朗读"
            }
        ]
    }


#定义函数进行故事数据格式转还
def story_summary(story: Story,include_content: bool=False) -> dict[str, Any]:
    #创建故事基础信息字典，用于返回概要信息
    item = {
        "id":story.id,
        "title":story.title,
        "category":story.category,
        "summary":story.summary,
        "words":story.words,
        "emoji":story.emoji,
        "coverType":story.cover_type,
        "date":story.created_at.isoformat(),
        "voices":["标准旁白"]
    }

    #判断是否需要返回完整的故事内容
    if include_content:
        item.update(
            {
                "content":story.content,
                "images":[image.url for image in story.images],
                "voices":voices()["voices"]
            }
        )

    return item