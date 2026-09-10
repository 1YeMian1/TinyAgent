#演示使用LangChain的Memory记忆，进行上下文的关联，模拟一个三轮的对话
#导入记忆存储对象
from  langchain_classic.memory import ConversationBufferMemory
#导入模型构建函数
from llm_factory import create_llm
#导入Schema三种消息类型
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

#定义函数，用来根据记忆和用户输入构造消息列表
def build_message(memory:ConversationBufferMemory ,user_input:str):
    #从记忆对象中获取消息列表
    history = memory.load_memory_variables({}).get("history",[])

    #返回包含三种消息对象的列表
    return  [
        #设置系统消息
        SystemMessage(
            content="你是一名LangChain 的课程讲师，只回答LangChain的相关问题"
        ),
        #设置用户消息
        HumanMessage(
            user_input
        ),
        #设置AI消息,将历史消息展开到列表中
        *history,
    ]


#定义一个带记忆的函数
def ask_with_memory(memory:ConversationBufferMemory,user_input:str)->str:

    #1.构建模型对象
    llm = create_llm()

    #2.根据用户的输入和记忆信息，调用函数构造消息列表
    message = build_message(memory,user_input)

    #3.调用模型执行
    response = llm.invoke(message)

    #4.取出模型返回的文本消息
    answer = response.content

    #5.将本轮的用户输入和AI返回的信息，保存到记忆中
    memory.save_context({"input":user_input},{"output":answer})

    return answer


#定义main函数执行
def main()->None:
    #1.构建一个用于对轮对话的缓存记忆对象
    memory =ConversationBufferMemory(
        #设置记忆变量在提示词中的名称
        memory_key="history",
        #设置用户输入的字段名称
        input_key="input",
        #设置模型输出的字段名称
        output_key="output",
        #设置返回返回历史记录时使用消息对象的形式
        return_messages=True
    )

    #准备一个三轮的固定问题列表
    questions = [
        #第一轮对话
        "你好，我正在学习LangChain框架的使用",
        #第二轮 对话
        "请给我解释一下Memory 是什么，其中都有哪些记忆对象，作用是什么？",
        #第三轮对话
        "你还记得我刚才在学习什么吗？",
    ]

    #遍历问题列表
    for index , input  in enumerate(questions,start=1):

        print(f"=================== 第{index}轮 ====================")
        print(f"学生：{input}")

        #调用带记忆的函数，获取对应的回答
        answer = ask_with_memory(memory,input)

        print(f"助手：{answer}")


if __name__ == "__main__":
    main()