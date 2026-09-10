"""SQLAlchemy persistence models."""

# 开启未来注解支持，允许类型延迟解析
from __future__ import annotations

# 导入日期时间相关模块
from datetime import datetime, timezone

# 导入 SQLAlchemy 字段类型和约束
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint

# 导入 ORM 映射相关方法
from sqlalchemy.orm import Mapped, mapped_column, relationship

# 导入项目数据库基类
from backend.database import Base


# 定义 UTC 当前时间函数
def utc_now() -> datetime:
    # 返回当前 UTC 时间
    return datetime.now(timezone.utc)


# 用户数据模型
class User(Base):
    # 指定数据库表名
    __tablename__ = "users"

    # 用户主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 用户名称
    username: Mapped[str] = mapped_column(String(80), nullable=False)

    # 用户邮箱，唯一索引
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)

    # 用户密码加密值
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


# 故事数据模型
class Story(Base):
    # 指定故事表名
    __tablename__ = "stories"

    # 故事主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 故事标题
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # 故事分类
    category: Mapped[str] = mapped_column(String(80), index=True, nullable=False)

    # 故事简介
    summary: Mapped[str] = mapped_column(String(500), nullable=False, default="")

    # 故事正文内容
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 故事字数统计
    words: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 故事表情标识
    emoji: Mapped[str] = mapped_column(String(16), nullable=False, default="📖")

    # 封面类型
    cover_type: Mapped[str] = mapped_column(String(8), nullable=False, default="c1")

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # 关联故事图片，一对多关系
    images: Mapped[list["StoryImage"]] = relationship(
        # 设置反向关联字段
        back_populates="story",
        # 删除故事时同步删除图片
        cascade="all, delete-orphan",
        # 按图片 ID 排序
        order_by="StoryImage.id"
    )


# 故事图片模型
class StoryImage(Base):
    # 指定图片表名
    __tablename__ = "story_images"

    # 图片主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 关联故事 ID
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"), index=True, nullable=False)

    # 图片地址
    url: Mapped[str] = mapped_column(Text, nullable=False)

    # 图片生成提示词
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # 图片生成来源
    provider: Mapped[str] = mapped_column(String(80), nullable=False, default="local-illustration")

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # 关联故事对象
    story: Mapped[Story] = relationship(back_populates="images")


# 收藏记录模型
class Favorite(Base):
    # 指定收藏表名
    __tablename__ = "favorites"

    # 设置用户和故事联合唯一约束
    __table_args__ = (UniqueConstraint("user_id", "story_id", name="uq_favorite_user_story"),)

    # 收藏主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 用户 ID 外键
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 故事 ID 外键
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)

    # 收藏时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


# 浏览历史模型
class History(Base):
    # 指定历史记录表名
    __tablename__ = "history"

    # 历史记录主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 用户 ID 外键
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 故事 ID 外键
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)

    # 浏览时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)


# 智能助手会话模型
class ChatSession(Base):
    # 指定会话表名
    __tablename__ = "chat_sessions"

    # 会话主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 所属用户 ID 外键
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # 会话标题（默认取首条消息前 20 字）
    title: Mapped[str] = mapped_column(String(120), nullable=False, default="新会话")

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # 最近活跃时间（每次对话刷新）
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    # 关联消息，一对多关系
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


# 智能助手消息模型
class ChatMessage(Base):
    # 指定消息表名
    __tablename__ = "chat_messages"

    # 消息主键 ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 所属会话 ID 外键
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # 消息角色：user（用户）/ assistant（助手）
    role: Mapped[str] = mapped_column(String(20), nullable=False)

    # 消息正文
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 工具调用轨迹（JSON 字符串：[{tool, args, result}]）
    tool_steps: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # 引用来源（JSON 字符串：[{story_id, title}]）
    references: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # 运行模式：real（真模型）/ simulation（教学模拟）
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="simulation")

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    # 关联会话对象
    session: Mapped[ChatSession] = relationship(back_populates="messages")