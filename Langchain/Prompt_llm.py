#引入LangChain-core 下的Runnable接口对象，RunnableLambda对象
from langchain_core.runnables import RunnableLambda
#引入模型构建函数
from  llm_factory import create_llm
"""
Runnable接口: 所有的LangChain组件都是适配此接口，若需要将某个对象拼接到LangChain执行链中，
需要使用RunnableLambda进行包装，使其适配Runnable接口
"""
#定义函数，用来根据概念名称生成提示词
def  build_concept_prompt(concept:str)->str:
    return f"""
你是一名LangChain的框架课程的讲师，请只围绕LangChain框架的相关知识进行回答，解释此概念：
概念：{concept}
请使用Markdown格式进行输出：
# {concept}
## 1.{concept} 的简单解释
## 2.解释 {concept} 在LangChain中的作用
## 3.举一个简单案例
## 4.学习建议
""".strip()

#定义函数，调用模型解释概念
def explain_langchain_concept(concept:str)->str:

    #构建模型对象
    llm = create_llm()

    #使用RunnableLambda对象，包装提示构建函数，使其适配Runnable接口，进行执行链的拼接
    prompt_runnable = RunnableLambda(build_concept_prompt)

    #拼接执行链使用管道符“|”，将提示词构建函数和模型对象组装为执行；链
    chain = prompt_runnable | llm

    #执行执行链
    #invoke() 同步调用，将结果一次性输出
    result = chain.invoke(concept)

    #以文本形式返回执行结果
    return result.content

#定义函数测试执行
def demo_invoke()->None:
    print("=============== invoke() 普通调用 =================")
    #调用函数执行
    response = explain_langchain_concept("Runnable 和 RunnableLambda")
    #打印执行结果
    print(response)

#定义函数演示stream流式输出
def  demo_stream()->None:
    print("=============== stream() 流式调用 =================")
    #构建模型对象
    llm = create_llm()
    #使用RunnableLambda对象包装提示词构建构建函数使其适配runnable接口，进行链式组装
    prompt_runnable = RunnableLambda(build_concept_prompt)
    #使用管道符“|”组装执行链
    chain = prompt_runnable | llm
    #调用执行链
    response = chain.stream("Prompt")
    #遍历模型流式输出的每一段内
    for chunk in response:
        #逐块打印遍历的文本内容
        print(chunk.content,end="",flush=True)
    print()

def main()->None:
    demo_invoke()
    print("\n ----------------------------------------------------------------")
    demo_stream()

if __name__ == "__main__":
    main()








