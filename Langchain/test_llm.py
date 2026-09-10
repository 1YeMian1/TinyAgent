#基础模型调用
#导入模型对象
from llm_factory import create_llm

#定义函数进行测试
def main()->None:
    #创建一个可以直接调用的模型对象
    llm_model = create_llm()
    #准备一个用户问题
    question = "你好，我正在学习Langchain框架进行模型调研测试，解释一下LangChain框架的作用和组成。"
    #调用模型进行提问获取结果
    #invoke()  调用模型的同步函数，可以将返回的结果进行一次性的输出
    response = llm_model.invoke(question)
    #打印结果
    print(response.content)

if __name__ == "__main__":
    main()