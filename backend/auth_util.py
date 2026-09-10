#密码哈希加密和JWT身份认证
#开启注解支持
from __future__ import annotations
#哈希计算模块
import hashlib
#导入安全比较模型
import hmac
#随机安全数生成模块
import secrets
#日期模块
from datetime import datetime, timedelta, timezone
#导入可选类型
from typing import Optional
#JWT编码库
import jwt
#FastAPI依赖，异常，请求对象
from fastapi import Depends, HTTPException, status, Request
#引入Session会话
from sqlalchemy.orm import Session
#数据库依赖
from backend.database import get_db
#用户模型
from backend.models import User
#导入JWT参数
from backend.config import TOKEN_TTL_HOURS, JWT_SECRET, JWT_ALGORITHM


#定义哈希密码加密函数
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    #若没盐值，进行随机生成
    salt = salt or secrets.token_bytes(16)
    #使用PBKDF2算法生成密码摘要
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        210_000
    )
    #按固定格式保存算法，次数，盐值，摘要
    return f"pbkdf2_sha256$210000${salt.hex()}${digest.hex()}"


#定义函数创建JWT token
def create_token(user: User) -> str:
    #获取当前时间的时间戳
    now = datetime.now(timezone.utc)
    #生成并返回JWT字符串
    return jwt.encode(
        {
            #保存用户ID
            "sub": str(user.id),
            #设置签发时间
            "iat": now,
            #设置过期时间
            "exp": now + timedelta(hours=TOKEN_TTL_HOURS)
        },
        #使用秘钥签名
        JWT_SECRET,
        # 修复：复数 algorithms 列表参数
        algorithm=JWT_ALGORITHM
    )


#密码校验函数
def verify_password(password: str, encoded: str) -> bool:
    try:
        #拆分保存的密码哈希信息,算法，次数，盐值，摘要
        scheme, rounds, salt_hex, digits_hex = encoded.split("$")

        #检查算法是否一致
        if scheme != "pbkdf2_sha256":
            #若不一致校验失败返回False
            return False

        #使用相同的参数重新计算密码摘要
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            int(rounds)
        ).hex()

        #对比摘要信息
        digest_flag = hmac.compare_digest(actual, digits_hex)
        #若一致返回True
        return digest_flag

    #若出现异常或对比失败，捕获异常
    except (ValueError, TypeError):
        #返回False
        return False


#根据请求头中保存的JWT Token信息获取当前登录的用户信息
def optional_user(request: Request, db: Session) -> Optional[User]:
    #获取请求头中的Authorization字段
    header = request.headers.get("authorization")

    #是否携带Bearer Token
    if not header or not header.startswith("Bearer "):
        #没有token返回空用户
        return None

    #若有Token，解码Token捕获异常
    try:
        #解码
        payload = jwt.decode(
            header[7:],
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        #根据Token中的用户ID查询
        return db.get(User, int(payload["sub"]))

    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        return None


#获取当前登录用户信息
def current_user(
        request: Request,
        db: Session = Depends(get_db),
) -> User:
    #尝试获取当前用户
    user = optional_user(request, db)

    print("===================:", user)

    #判断用户是否登录，若获取不到，说明没有token信息未登录，返回401权限不足异常
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录"
        )

    return user
