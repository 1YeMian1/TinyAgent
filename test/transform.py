#导包,分词包
import  jieba

import thulac

str1 = "我现在正在学习大模型的实训课程，可以很熟练的使用Transform库，完成深度学习的模型开发内容。"

print("======================== jieba 分词   =========================")
#执行分词,使用列表接收分词后的结果
seg_list_jieba = list(jieba.cut(str1))
print(seg_list_jieba)

print("\n======================== thulac 分词   =========================")
#1.初始化
# seg_only=True 表示只输出分词结果，不加词性标注（结果更加清爽）
thu = thulac.thulac(seg_only=True)
#调用实例的cut方法
result_thu = thu.cut(str1,text=True)
print(result_thu)












