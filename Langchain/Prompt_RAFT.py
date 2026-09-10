#演示使用RAFT结构化的提示词，Schema三种消息类型
#引入三种消息类型
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
#引入runnable对象
from langchain_core.runnables import RunnableLambda
#引入模型构建对象
from llm_factory import create_llm


#定义函数，构建教师风格提示
def teacher_raft_prompt(question:str)->str:
   return f"""
R(Role 角色)  ：你是一名正在教授LangChain框架的计算机教师。
A(Action 任务)：请使用浅显易懂，由深入浅的方式进行LangChain基础内容教学，教学时保持严谨，清晰进行授课，解答此问题：{question}。
F(Format 格式)：请按照先讲定义，再讲作用，在使用课堂案例进行讲解说明，最后说明注意事项，四部分进行输出
T(Tone 语气)  ：严谨，清晰，结构化，适合课堂教学。
限制：请只围绕 LangChain 框架的相关内容进行讲解。
""".strip()

#定义函数，构建学生风格提示词
def student_raft_prompt(question:str)->str:
    return f"""
R(Role)   : 你是一名正在帮助同学学习LangChain框架使用的学习伙伴。  
A(Action) : 请使用最容易理解的语言解释LangChain的相关问题：{question}。
F(Format) : 请按"先说大白话,再说作用,生活类比,怎么练习" 四部分进行输出
T(Tone)   : 轻松，口语化，生活例子解释
限制：请只围绕 LangChain 框架的相关内容进行回答解释。
""".strip()


QUESTION_1 = "请解释LangChain中Runnable的作用？"
QUESTION_2 = "请解释LangChain中Schema三种消息类型是什么，作用是什么？"

#定义函数演示两种不同风格提示的效果
def run_raft_prompt_demo()->None:
    #构建模型对象
    print("=========================== 1.构建模型对象 ===============================")
    llm = create_llm()
    #使用runnableLambda对象对构建两中不同风格提示词的函数进行包装，使其适配runnable接口
    print("=========================== 2.封装runnable对象 ===============================")
    teacher_prompt_runnable = RunnableLambda(teacher_raft_prompt)
    student_prompt_runnable = RunnableLambda(student_raft_prompt)
    #使用管道符"|"，使用模型对象和提示词runnable对象组装执行链
    print("=========================== 3.组装执行链 ===============================")
    teacher_chain = teacher_prompt_runnable | llm
    student_chain = student_prompt_runnable | llm
    #调用执行链执行
    print("=========================== 4.调用教师风格Prompt执行链 ===============================")
    teacher_response = teacher_chain.invoke(QUESTION_1)
    print(teacher_response.content)
    print("=========================== 5.调用学生风格Prompt执行链 ===============================")
    student_response = student_chain.stream(QUESTION_2)
    #遍历每次输出的片段
    for chunk  in student_response:
        print(chunk.content,end="",flush=True)
    print()


#定义main函数执行
def main():
    run_raft_prompt_demo()

if __name__ == "__main__":
    main()









