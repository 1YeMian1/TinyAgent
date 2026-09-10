#开启注解支持
from __future__ import annotations

#支持导入JSON数据处理模块
import json

#导入生成器类型
from  typing import Generator

#SQLAlchemy 核心组件
from  sqlalchemy import create_engine ,select

#导入ORM基类
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

#导入配置类
from backend.config import DATABASE_URL,DATASET_PATH,EMOJIS



#1.创建ORM数据模型基类
class Base(DeclarativeBase):
    #空实现，作为所有的实体的父类
    pass

#2.创建SQLite数据库连接引擎
engine = create_engine(
    #数据库连接地址
    DATABASE_URL,
    #关闭SQLLit线程检查
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

#3.创建数据库会话工厂
SessionLocal = sessionmaker(
    #绑定数据库引擎
    bind=engine,
    #禁止自动提交事务
    autoflush=False,
    #提交后保持对象状态
    expire_on_commit=False,
)


#4.定义数据库依赖获取方法
def get_db()->Generator[Session, None, None]:

    #创建数据库会话
    db = SessionLocal()

    #捕获请求执行过程
    try:
        #返回数据库会话提供给外部使用
        yield db

    finally:
        #资源释放
        db.close()

#初始化数据库并填充故事数据
def seed_database()->None:
    #导入模型，避免循环使用
    from backend.models import Story

    #根据ORM模型创建数据表
    Base.metadata.create_all(engine)

    #创建数据库会话
    with SessionLocal() as db:
        #检测数据库中是否已有故事数据
        exits = db.scalar(
            select(Story.id).limit(1)
        )

        #已存在数据直接结束
        if exits is not None:
            return

        try:
            #若不存在，读取故事JSON文件
            raw = json.loads(
                DATASET_PATH.read_text(
                    encoding="utf-8"
                )
            )
        except (FileNotFoundError,
                json.JSONDecodeError):
            #读取文件异常时使用空数据列表
            raw = []

        #循环遍历故事数据集
        for index , item in enumerate(raw,1):
            #获取分类信息
            category = (
                    item.get("category")
                    or "想象故事"
                )

            #获取故事的文本
            content =(
                    item.get("text")
                    or ""
            )

            #若内容为空进行跳过
            if not content:
                continue

            #根据正文生成标题
            title = (
                content[:24].split("。",1)[0]
                or f"{category}#{index}"
            )

            #创建故事数据库对象
            story_info = Story(
                #设置故事编号
                id=(item.get("id") or index),
                #设置标题
                title=title,
                #设置分类
                category=category,
                #设置故事简介
                summary=content[:100],
                #设置完整内容
                content=content,
                #设置字数
                words=len(content),
                #设置图标
                emoji=EMOJIS.get(category,"📚"),
                #设置封面类型
                cover_type=f"c{(index % 7 ) + 1}"
            )
            #添加故事对象到会话
            db.add(story_info)
        #提交事务
        db.commit()
















