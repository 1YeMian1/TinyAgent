#使用Vit 实现图片分类任务
#导入 Vit 图片预处理器
from transformers import ViTImageProcessor

#导入 Vit 图片分类模型
from transformers import ViTForImageClassification

#导入图片处理库
from PIL import Image
#导入HTTP 请求库
import  requests

#加载图片预处理模型
pretrained = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

#加载图片分类预训练模型
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
)


#定义待分类的图片地址
img1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"


#请求图片对象转换为 PIL 图片对象
img1 = Image.open(
    requests.get(img1,stream=True).raw
)

#图片预处理工作，使用Vit处理图片，将图片转换为模型输入格式
#类似NLP中tokenizer 中的将文本转换为token
model_input=pretrained(
    images=img1,
    return_tensors="pt",
)

#输出模型输入的数据结构
print(model_input)


#模型推理
#将图片输入模型进行预测
output = model(
    **model_input,
)

#获取预测结果的索引
oid = output.logits.argmax(-1).item()

#根据索引获取对应的类别名称
print(
    "预测结果：",
    model.config.id2label[oid]
)

#Top-5 结果展示
#对所有的类别概率进行归一化，并获取概率最高的五个类别
top5 = output.logits.softmax(-1).topk(5)

print("Top-5 预测：")

#遍历预测概率和类别索引
for prob ,  idx  in zip(
    top5.values[0],
    top5.indices[0]
):
    print(
        f"{model.config.id2label[idx.item()]:20s}"
        f"{prob.item():.4f}"
    )

