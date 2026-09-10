#导入魔搭社区模型推理接口
from modelscope.pipelines import pipeline

#导入任务类型对象
from modelscope.utils.constant import Tasks

#导入图片读取和绘制工具
from  PIL import Image,ImageDraw


#加载模型 DAMO-YOLO目标检测模型
detector = pipeline(
    Tasks.image_object_detection,
    model="iic/cv_tinynas_object-detection_damoyolo",
    trust_remote_code=True
)

#定义待检测的目标对象路径
image_path = r"D:\QF\Agent\Tiny_Agent\images\cat.jpg"

#打开图片，转换图片格式
image = Image.open(image_path).convert("RGB")

#调用模型对象进行目标检测
result = detector(image_path)

#输出模型的原始结果
print(result)

#创建图片绘制工具
draw = ImageDraw.Draw(image)

print("\n=================== 检测结果 ====================")

#遍历检测结果
for box , label ,score  in zip(
    result["boxes"],
    result["labels"],
    result["scores"],
):
    #过滤置信度过低的目标
    if score < 0.5:
        continue

    #输出目标类别，置信度，坐标
    print(
        f"类别：{label:15s}"
        f"置信度：{score:.3f}"
        f"坐标：{box}"
    )

    #获取检测框
    x1,y1,x2,y2 = box

    #在图片中绘制目标检测框
    draw.rectangle(
        [x1,y1,x2,y2],
        outline="red",
        width=3,
    )

    #在检测框上显示类别，置信度
    draw.text(
        (x1,y1),
        f"{label}:{score:.2f}",
        fill="green",
    )

#保存绘制完检测框的图片
image.save("./detection_result.jpg")


print("\n=========================检测完成，结果保存：detection_result.jpg=========================")







