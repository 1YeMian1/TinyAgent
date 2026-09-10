# 标识当前文件功能：故事生成、SSE流式输出、图片生成、语音接口
"""Story, SSE, image, and voice generation endpoints."""

# 启用未来版本类型注解兼容
from __future__ import annotations

# 导入JSON处理模块
import json
# 导入类型定义工具
from typing import Any, Generator

# 导入FastAPI路由、依赖注入、请求参数工具
from fastapi import APIRouter, Depends, Query, Request
# 导入流式响应对象
from fastapi.responses import StreamingResponse
# 导入数据库Session
from sqlalchemy.orm import Session

# 导入用户认证相关方法
from backend.auth_util import current_user, optional_user
# 导入配置变量
from backend.config import EMOJIS, MODEL_PATH
# 导入数据库获取方法
from backend.database import get_db
# 导入数据库模型
from backend.models import History, Story, User
# 导入请求数据模型
from backend.schemas import GenerateRequest, ImagesRequest, VoiceRequest
# 导入故事序列化方法
from backend.serializers import story_summary
# 导入内容生成服务
from backend.services.generator_v2 import generate_content, normalize_category
# 导入图片生成服务
from backend.services.image_generator import generate_story_images


# 创建生成相关API路由
router = APIRouter(prefix="/api/v1/generate", tags=["generation"])


# 定义故事生成接口
@router.post("/story")
# 接收故事生成请求并返回结果
def generate_story(payload: GenerateRequest, request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    # 标准化故事分类
    category = normalize_category(payload.category, payload.theme)
    # 更新请求中的分类字段
    payload = payload.model_copy(update={"category": category})
    # 调用生成模型生成故事内容
    content, generator = generate_content(payload)
    # 获取角色名称，没有则使用默认名称
    character = payload.character.strip() or "小主角"
    # 创建故事数据库对象
    story = Story(
        title=f"{payload.theme.strip()}：{character}的故事",
        category=category,
        summary=content[:100],
        content=content,
        words=len(content),
        emoji=EMOJIS.get(category, "✨"),
        cover_type="c7"
    )
    # 添加故事记录到数据库
    db.add(story)
    # 获取生成后的故事ID
    db.flush()
    # 获取当前登录用户（允许匿名）
    user = optional_user(request, db)
    # 初始化图片服务商变量
    image_provider = None
    # 判断是否需要生成图片并且用户已登录
    if payload.generate_images and user:
        # 调用图片生成服务
        image_urls, image_provider = generate_story_images(story, payload.image_count)
        # 导入故事图片模型
        from backend.models import StoryImage
        # 保存生成的图片记录
        db.add_all([StoryImage(story_id=story.id, url=url, provider=image_provider) for url in image_urls])
    # 判断用户是否存在
    if user:
        # 保存用户浏览历史
        db.add(History(user_id=user.id, story_id=story.id))
    # 提交数据库事务
    db.commit()
    # 刷新故事对象数据
    db.refresh(story)
    # 转换故事输出格式
    result = story_summary(story, include_content=True)
    # 返回实际使用的生成器
    result["generator"] = generator
    # 判断模型是否配置
    result["model_configured"] = bool(MODEL_PATH)
    # 返回图片生成服务商
    result["images_provider"] = image_provider
    # 未登录提示图片功能
    result["images_message"] = "登录后可生成并保存配图" if payload.generate_images and not user else None
    # 返回最终结果
    return result


# 定义故事流式输出接口
@router.get("/story/stream")
# 使用SSE方式持续返回故事内容
def stream_story(
    theme: str,
    category: str = "想象故事",
    character: str = "小主角",
    length: str = Query("medium", pattern="^(short|medium|long)$"),
    extra: str = ""
) -> StreamingResponse:
    # 调用生成服务生成完整故事
    content, _ = generate_content(
        GenerateRequest(theme=theme, category=category, character=character, length=length, extra=extra)
    )

    # 定义事件生成器
    def events() -> Generator[str, None, None]:
        # 按24字符切分内容
        for chunk in (content[index:index + 24] for index in range(0, len(content), 24)):
            # 返回SSE格式数据
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        # 返回结束事件
        yield 'data: {"done": true}\n\n'

    # 返回流式响应
    return StreamingResponse(events(), media_type="text/event-stream")


# 定义图片重新生成接口
@router.post("/images")
# 接收故事ID并生成图片
def generate_images(
    payload: ImagesRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    # 当前用户仅用于权限验证
    del user
    # 导入图片模型
    from backend.models import StoryImage

    # 根据ID查询故事
    story = db.get(Story, payload.story_id)
    # 判断故事是否存在
    if not story:
        # 导入异常类
        from fastapi import HTTPException
        # 返回404错误
        raise HTTPException(status_code=404, detail="故事不存在")

    # 调用图片生成服务
    image_urls, provider = generate_story_images(story, payload.count)
    # 清除旧图片关联
    story.images.clear()
    # 保存新生成图片
    db.add_all([StoryImage(story_id=story.id, url=url, provider=provider) for url in image_urls])
    # 提交数据库修改
    db.commit()
    # 返回图片结果
    return {
        "images": image_urls,
        "available": True,
        "provider": provider,
        "count": len(image_urls)
    }


# 定义语音生成接口
@router.post("/voice")
# 返回语音生成状态
def generate_voice(
    payload: VoiceRequest,
    user: User = Depends(current_user)
) -> dict[str, Any]:
    # 当前用户仅用于权限验证
    del user
    # 返回浏览器语音合成提示
    return {
        "available": False,
        "voice_id": payload.voice_id,
        "message": "使用浏览器语音合成，请在故事详情页播放"
    }