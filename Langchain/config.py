#读取Langchain配置
#引入dataclass装饰器
from dataclasses import dataclass
#os模块，读取环境变量
import  os
#导入dotenv 模块load_dotenv 加载配置文件
from dotenv import load_dotenv


#1.使用装饰器定义配置对象
@dataclass(frozen=True)
class Settings:
    #配置平台API接口
    openai_base_url:str
    #配置api-key
    openai_api_key:str
    #配置模型名称
    openai_model:str

#2.定义函数读取配置信息
def get_settings()->Settings:
    #加载配置文件
    load_dotenv()
    #读取配置信息,平台接口，api秘钥，模型名称
    openai_base_url = os.getenv("OPENAI_BASE_URL","").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY","").strip()
    openai_model = os.getenv("OPENAI_MODEL","").strip()
    #创建空列表，用来保存缺失配置项名称
    missing = []
    #若未读取到平台接口信息将变量名称追加到列表中
    if not openai_base_url:
        missing.append("OPENAI_BASE_URL")
    #若未读取到模型秘钥信息，将变量名称追加到列表中
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    #若未读取到模型名称，将其追加到列表
    if not openai_model:
        missing.append("OPENAI_MODEL")

    #若列表不为空，说明有缺失的配置信息，返回异常信息
    if missing:
        #将缺失项信息拼接为字符串
        missing_text = ",".join(missing)
        #抛出异常
        raise ValueError(
            "缺失模型配置："f"{missing_text}。\n"
        )

    #若配置信息完整,封装配置对象返回配置
    return Settings(
        # 配置平台API接口
        openai_base_url=openai_base_url,
        # 配置api-key
        openai_api_key=openai_api_key,
        # 配置模型名称
        openai_model=openai_model,
    )
