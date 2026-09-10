# ============================================================
# 使用阿里魔搭社区 ModelScope 的 ViT 实现图片分类任务
# ============================================================


# 导入魔搭模型推理接口
from modelscope.pipelines import pipeline

# 导入任务类型
from modelscope.utils.constant import Tasks

# 导入图片处理库
from PIL import Image

# 导入HTTP请求库
import requests



# ============================================================
# 1. 加载魔搭社区 ViT 图片分类模型
# ============================================================

model = pipeline(
    Tasks.image_classification,
    model="damo/cv_vit-base_image-classification_imagenet1k"
)



# ============================================================
# 2. 定义待分类图片地址
# ============================================================

img_url = (
    "https://huggingface.co/datasets/"
    "huggingface/documentation-images/"
    "resolve/main/beignets-task-guide.png"
)



# ============================================================
# 3. 下载图片并转换为PIL对象
# ============================================================

image = Image.open(
    requests.get(
        img_url,
        stream=True
    ).raw
)



# 查看图片信息
print("图片信息:")
print(image)



# ============================================================
# 4. 模型推理
# ============================================================

result = model(
    image
)



# ============================================================
# 5. 输出预测结果
# ============================================================

print("\n预测结果:")

for item in result:
    print(
        "类别:",
        item["label"],
        "概率:",
        round(item["score"],4)
    )
