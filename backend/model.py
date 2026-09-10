#模型对象文件，管理模型对象，
# 基于ORM(Object Relational Mapping)-通过模型或者实体映射数据库表中的字段，
# 一张表对应一个实体或者模型对象，表中的字段信息和模型中的属性信息一一对应

#开启注解支持
from __future__ import annotations

#导入日期时间模型
from datetime import datetime,timezone

#导入sqlAlchemy 中的字段类型和约束
from sqlalchemy import DateTime,ForeignKey,Integer,String,Text,UniqueConstraint

#导入ORM映射的相关方法
from sqlalchemy.orm import Mapped,mapped_column,relationship

#导入基类
from backend.database import Base

#定义UTC当前时间函数
def utc_now()->datetime:
    return datetime.now(timezone.utc)


#1.构建用户数据模型
class User(Base):
    #指定数据库所关联的表
    __tablename__ = "users"

    #映射字段
    #主键ID
    id:Mapped[int] = mapped_column(Integer,primary_key=True)

    #用户名称
    username:Mapped[str] = mapped_column(String(80),nullable=False)

    #邮箱
    email:Mapped[str] = mapped_column(String(255),unique=True,nullable=False,index=True)

    #密码
    password_hash:Mapped[str] = mapped_column(String(255),nullable=False)

    #创建时间
    created_at:Mapped[datetime] = mapped_column(DateTime,nullable=False,default=utc_now)

#2.构建故事模型
class Story(Base):
    #指定所关联表名称
    __tablename__ = "stories"

    #主键ID
    id:Mapped[int] = mapped_column(Integer,primary_key=True)
    #故事标题
    title:Mapped[str] = mapped_column(String(255),nullable=False)
    #故事分类
    category:Mapped[str] = mapped_column(String(80),nullable=False)
    #故事摘要
    summary:Mapped[str] = mapped_column(String(500),nullable=False,default="")
    #故事正文
    content:Mapped[str] = mapped_column(Text,nullable=False)
    #字数统计
    words:Mapped[int] = mapped_column(Integer,nullable=False,default=0)
    #表情标识
    emoji:Mapped[str] = mapped_column(String(16),nullable=False,default="🏷️")
    #封面类型
    cover_type:Mapped[str] = mapped_column(String(8),nullable=False,default="c1")
    #创建时间
    created_at:Mapped[datetime] = mapped_column(DateTime,nullable=False,default=utc_now)

    #关联故事图片
    images:Mapped[list["StoryImage"]] = relationship(
        #设置反向关联字段
        back_populates="story",
        #删除故事时同步删除图片
        cascade="all,delete-orphan",
        #按图片ID排序
        order_by="StoryImage.id"
    )

#3.构建故事图片模型
class StoryImage(Base):
    #指定关联表名称
    __tablename__ = "story_images"

    #主键id
    id:Mapped[int] = mapped_column(Integer,primary_key=True)
    #关联故事ID,外键，删除关联信息时进行同步删除
    story_id:Mapped[int] = mapped_column(Integer,ForeignKey("stories.id",ondelete="CASCADE"),index=True,nullable=False)
    #保存图片的url
    url:Mapped[str] = mapped_column(Text,nullable=False)
    #prompt提示词
    prompt:Mapped[str] = mapped_column(Text,nullable=False,default="")
    #图片生成来源
    provider:Mapped[str] = mapped_column(String(80),nullable=False,default="local_illustration")
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    #关联story表故事对象
    story:Mapped[Story] = relationship(back_populates="images")

#4.收藏表
class Favorite(Base):
    #指定所关联的表名称
    __tablename__ = "favorites"
    #设置用户和故事联合的唯一约束
    __table_args__ = (UniqueConstraint("user_id","story_id",name="uq_favorite_user_story"),)
    #主键ID
    id:Mapped[int] = mapped_column(Integer,primary_key=True)
    #用户ID
    user_id:Mapped[int]  = mapped_column(ForeignKey("users.id",ondelete="CASCADE"),nullable=False)
    #故事ID
    story_id:Mapped[int] = mapped_column(ForeignKey("stories.id",ondelete="CASCADE"),nullable=False)
    #收藏时间
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


#5.浏览历史
class History(Base):
    #关联的表名称
    __tablename__ = "history"
    # 主键ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 用户ID
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # 故事ID
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id", ondelete="CASCADE"), nullable=False)
    # 浏览时间
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)



