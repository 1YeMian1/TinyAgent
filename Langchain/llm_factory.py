#使用工厂模型思想，构建模型对象
#导入ChatOpenAI对象
from langchain_openai import ChatOpenAI
#导入读取配置函数
from config import get_settings


#定义函数构建模型对象
#temperature 参数，用于控制模型返回信息的发散性和创造性程度，值越高，发散性越高，返回的信息越不准确
def create_llm(temperature:float=0.7)->ChatOpenAI:
    #获取配置对象
    settings = get_settings()
    #封装模型对象进行返回
    return ChatOpenAI(
        #配置模型名称
        model=settings.openai_model,
        #配置模型秘钥
        api_key=settings.openai_api_key,
        #配置模型平台接口url
        base_url=settings.openai_base_url,
        #配置模型温度参数
        temperature=temperature,
    )