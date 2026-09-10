#注册，登录以及当前用户的相关接口模快
#通用类型支持
from  typing import Any

#导入FastAPI路由依赖注入和异常处理
from fastapi import APIRouter,Depends,HTTPException

#SQL查询构造器
from sqlalchemy import select,func

#会话对象
from sqlalchemy.orm import Session

#导入数据库依赖注入
from backend.database import get_db

#导入token密码加密，密码验证函数
from backend.auth_util import hash_password, create_token, verify_password, current_user

#导入模型
from backend.models import User, Favorite, History

#请求和响应数据模型
from backend.schemas import AuthRequest,RegisterRequest,UserResponse


#创建认证模块路由
router = APIRouter(prefix="/api/v1/auth",tags=["auth"])

#定义注册接口
@router.post("/register")
def register(
        #接收注册请求参数
        pyload:RegisterRequest,

        #注入数据库会话
        db:Session = Depends(get_db)
)-> dict[str,Any]:

    #将邮箱转换为小写
    email = pyload.email.lower()

    #进行邮箱唯一性校验
    user_email = db.scalar(select(User).where(User.email == email))

    #校验
    if user_email:
        raise HTTPException(
            status_code=409,
            detail="该邮箱已注册"
        )

    #创建用户对象
    user = User(
        #设置用户名,去除空格
        username=pyload.username.strip(),
        #设置邮箱
        email=email,
        #设置密码进行哈希加密
        password_hash=hash_password(pyload.password)
    )

    #进行数据添加
    db.add(user)
    #提交事务
    db.commit()
    #刷新数据库
    db.refresh(user)

    #返回用户信息和登录的token
    return {
        #返回用户响应模型
        "user":UserResponse.model_validate(user),
        #创建并返回JWT Token
        "token":create_token(user)
    }


#用户登录
@router.post("/login")
def login(
        payload : AuthRequest,
        db:Session = Depends(get_db)
) -> dict[str,Any]:
    #根据用户的邮箱信息查询用户数据
    user = db.scalar(
        select(User).where(User.email == payload.email.lower())
    )

    #校验若用户不存在或者密码校验失败
    if not user or not verify_password(
            payload.password,
            user.password_hash
        ):
        #校验失败返回异常信息
        raise HTTPException(
            #401 权限不足异常
            status_code=401,
            detail="邮箱或密码错误"
        )

    #登录成功返回用户信息和登录凭证
    return {
        #返回用户信息
        "user":UserResponse.model_validate(user),
        #返回JWT凭证
        "token":create_token(user),
    }


#获取用户信息
@router.get("/profile")
def profile(
        user:User = Depends(current_user),
        db:Session = Depends(get_db)
)->dict[str,Any]:
    #查询用收藏的故事数量
    favorite_count = db.scalar(
        select(func.count())
        .select_from(Favorite)
        .where(Favorite.user_id == user.id)
    ) or 0

    #查询用户生成或者浏览故事数量
    history_count = db.scalar(
        select(func.count())
        .select_from(History)
        .where(History.user_id == user.id)
    )or 0

    return {
        #返回用户基本信息
        "user":UserResponse.model_validate(user),
        #返回用户数据统计
        "stats":{
            "favorites":favorite_count,
            "generated":history_count,
            "listened":0
        }


    }







