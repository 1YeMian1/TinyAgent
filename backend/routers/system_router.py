#当模块提供系统相关接口，例如分类信息查询
#导入通用类型注解Any
from typing import Any


from sqlalchemy import select
#导入FastAPI路由管理器
from fastapi import APIRouter,Depends

#引入会话对象
from sqlalchemy.orm import Session

#导入系统分类配置资源
from backend.config import CATEGORIES 
#引入模型对象
from backend.models import User,Favorite,History,Story
#引入数据格式化函数
from  backend.serializers import  story_summary
#引入数据源
from backend.database import get_db

from backend.auth_util import current_user

#创建接口路由对象，设置用意的访问前缀和接口分组
router = APIRouter(
    prefix="/api/v1",
    tags=["system"]
)

#定义GET请求接口，用于获取分类信息
@router.get("/categories")
def categories() -> dict[str, Any]:
    """
    获取系统支持的故事分类列表及全局 Emoji 映射。
    提供给前端分类 Tab 栏与创作选择框使用。
    """
    from backend.config import EMOJIS
    return {
        "categories": CATEGORIES,
        "emojis": EMOJIS,
        "total": len(CATEGORIES)
    }



