#当前模块提供故事信息列表查询，详细信息查询，收藏、历史信息查询。
#开启注解支持
from __future__ import annotations
#通用类型注解,可选的Optional
from typing import Any,Optional
#FastAPI路由，依赖，异常处理，查询参数工具
from fastapi import APIRouter,Depends,HTTPException,Query
#导入sqlAlchemy 聚合函数，查询工具
from sqlalchemy import func,or_,select
#数据库会话
from sqlalchemy.orm import Session
#导入数据库模型
from backend.models  import Story,User,Favorite,History
#导入数据库🔗获取
from backend.database import get_db
#导入序列化数据转换工具
from  backend.serializers import story_summary

from backend.auth_util import current_user

#创建故事模块路由，指定前缀和模块信息
router = APIRouter(prefix="/api/v1/stories",tags=["stories"])

#定义函数查询故事信息列表
@router.get("")
def list_stories(
        #接收可选的故事分类筛选参数
        category:Optional[str] = None,
        #接收可选的故事关键字搜索参数
        search : Optional[str] = None,
        #设置分页参数，查询第几页
        page : int = Query(1,ge=1),
        #设置每页查询多少条数据
        page_size : int = Query(24,ge=1,le=100),
        #通过依赖注入获取数据库会话对象
        db : Session = Depends(get_db)
)->dict[str,Any]:

    #构建查询列表，用于动态的查询条件拼接
    filters = []

    #校验参数,判断是否存在有效的分类参数
    if category and category != "undefined":
        #若存在则进行条件设置
        filters.append(Story.category == category)

    #判断关键字参数
    if search and search != "undefined":
        #若参数存在,构建SQL模糊查询关键字
        term = f"%{search}%"

        #添加标题，摘要，正文内容的模糊匹配条件
        filters.append(
            or_(
                Story.title.like(term),
                Story.summary.like(term),
                Story.content.like(term),
            )
        )

    #查询满足条件的故事总数，用于分页显示
    total = db.scalar(
        select(func.count())
        .select_from(Story)
        .where(*filters)
    ) or 0

    #创建分页查询语句
    query=(
        select(Story)
        .where(*filters)
        .order_by(Story.id)
        .offset((page-1)*page_size)
        .limit(page_size)
    )

    #执行查询获取当前页故事信息列表
    story_list = db.scalars(query).all()

    return {
        "stories": [story_summary(story) for story in story_list],
        "total":total,
        "page":page,
        "page_size":page_size,
    }


#根据ID获取故事详情
@router.get("/{story_id:int}")
def get_story_info(
        story_id : int,
        db : Session = Depends(get_db)
)->dict[str,Any]:
    #根据传入的ID查询数据中的信息
    story = db.get(Story,story_id)
    #非空校验
    if not story:
        #若未查询到数据信息抛出异常
        raise  HTTPException(
            status_code=404,
            detail="故事不存在"
        )

    return story_summary(story,include_content=True)



#获取用户故事信息列表
def _user_stories(user:User ,db:Session,favorite_mode:bool)->dict[str,Any]:
    #判断当前查询模式是否为收藏列表
    if favorite_mode:

        query = (
            #构建查询语句
            select(Story)
            #关联收藏表，通过story_id进行匹配故事信息
            .join(Favorite,Favorite.story_id == Story.id)
            #筛选当前用户的收藏记录，通过user_id进行匹配查询
            .where(Favorite.user_id == user.id)
            #根据创建时间进行降序排序
            .order_by(Favorite.created_at.desc())
        )

    else:
        #若查询的不是收藏信息，则创建查询历史浏览故事语句
        query = (
            select(Story)
            #关联历史记录表，通过story_id匹配查询
            .join(History,History.story_id == Story.id)
            #关联用户ID进行条件匹配
            .where(History.user_id == user.id)
            #根据创建时间进行降序排序
            .order_by(History.created_at.desc())
        )

    # 执行查询获取故事列表
    stories = db.scalars(query).all()

    # 返回结果
    return {
        "stories": [
            story_summary(story)
            for story in stories
        ],
        "total": len(stories)
    }

#查询收藏信息
@router.get("/favorites")
def get_Favorites(
        user:User  = Depends(current_user),
        db:Session = Depends(get_db)
)->dict[str,Any]:
    #查询收藏信息
    return _user_stories(user,db,True)


# 查询历史信息
@router.get("/history")
def get_History(
        user: User = Depends(current_user),
        db: Session = Depends(get_db)
) -> dict[str, Any]:
    # 查询历史信息
    return _user_stories(user, db, False)

@router.post("/{story_id:int}/favorite")
def toggle_favorite(story_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
    if not db.get(Story, story_id):
        raise HTTPException(status_code=404, detail="故事不存在")
    favorite = db.scalar(select(Favorite).where(Favorite.user_id == user.id, Favorite.story_id == story_id))
    if favorite:
        db.delete(favorite)
        favorited = False
    else:
        db.add(Favorite(user_id=user.id, story_id=story_id))
        favorited = True
    db.commit()
    return {"favorited": favorited, "id": story_id}


