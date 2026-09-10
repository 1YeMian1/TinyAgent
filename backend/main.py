#FastAPI应用
#开启注解支持
from  __future__ import annotations
#导入异步生命周期管理工具
from contextlib import asynccontextmanager
#异步生成器类型
from typing import AsyncGenerator
#FastAPI核心框架
from fastapi import FastAPI
#跨域请求中间件
from fastapi.middleware.cors import CORSMiddleware
#静态文件服务组件
from fastapi.staticfiles import StaticFiles
#导入跨域配置和静态文件目录
from backend.config import  CORS_ORIGINS,STATIC_DIR
#导入数据库初始化函数
from backend.database import seed_database

#引入路由模块
from backend.routers import system_router, storeis_router, auth_router,assistant,generate


#使用异步上下文管理器管理应用生命周期
@asynccontextmanager
#定义异步函数定义FastAPI应用启动和关闭的逻辑
async def lifespan(_: FastAPI)-> AsyncGenerator[None,None]:
    #应用启动时进行数据库初始化
    seed_database()

    #暂停生命周期管理，使应用进入正常运行状态
    yield

#创建FastAPI应用实例
app = FastAPI(
    #设置接口文档显示的项目名称
    tilte = "TinyLLM-Story API",
    #设置当前API服务版本
    version = "v1.0.0",
    #注册应用生命周期管理函数
    lifespan=lifespan
)


#添加跨域资源共享中间件
app.add_middleware(
    #指定使用CORS跨域处理中间件
    CORSMiddleware,
    #设置允许访问后端接口的来源地址
    allow_origins=CORS_ORIGINS,
    #设置是否允许携带Cookie等认证信息
    allow_credentials="*" not in CORS_ORIGINS,
    #设置允许的HTTP请求方法
    allow_methods=["*"],
    #设置允许的请求头信息
    allow_headers=["*"]
)

#注册路由
app.include_router(system_router.router)
app.include_router(storeis_router.router)
app.include_router(auth_router.router)
app.include_router(assistant.router)
app.include_router(generate.router)




#判断当前前端静态资源目录是否存在
if STATIC_DIR.exists():
    #挂载静态资源文件服务
    app.mount(
        #设置惊天资源访问的根路径
        "/",
        #指定静态文件目录并支持HTML页面访问
        StaticFiles(directory=STATIC_DIR,html=True),
        #设置静态资源服务名称
        name="fronted"
    )


