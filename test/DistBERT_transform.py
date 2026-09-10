#基于DitsBERT实现文本情感分析
#设置HuggingFace国内镜像地址，避免国外模型下载失败
import  os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

#导入系统文件操作，以及深度学习、数组处理 库
import  shutil
import torch
import numpy as np

#导入数据集加载工具
from datasets import load_dataset

#导入Transform组件
from  transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

#导入模型的评价指标
import evaluate


#定义main()
def main():

    print("=================== 1.数据集加载 ===================")
    dataset = load_dataset("stanfordnlp/imdb")

    #获取训练数据和测试数据
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    #查看基本数据信息
    print(dataset)
    print(train_dataset[0])


    print("=================== 2.Tokenizer 编码 -> TokenID ===================")
    #指定所使用模型信息
    model_name = ("distilbert/distilbert-base-uncased")

    #加载文本分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #定义文本编码函数
    def tokenize(example):
        #将文本转换为Token序列
        return  tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding=True,
        )

    #对训练集进行token编码
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    #对测试数据进行token编码
    test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    print("=================== 3.Padding 格式整理 ===================")
    #创建动态padding工具
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("=================== 4.加载预训练模型 ===================")
    #模型加载
    model = AutoModelForSequenceClassification.from_pretrained(
        #指定模型名称
        model_name,

        #标签数量
        num_labels=2,

        #设置类别名称映射
        id2label={
            0:"NEGATIVE",
            1:"POSITIVE"
        },

        #设置类编号映射
        label2id={
            "NEGATIVE" : 0,
            "POSITIVE" : 1
        }

    )

    print("=================== 5.配置模型训练参数 ===================")
    training_args = TrainingArguments(
        #模型的输出目录
        output_dir="./output",

        #设置训练轮数
        num_train_epochs=3,

        #设置训练和验证的Batch大小
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        #设置学习率
        learning_rate=2e-5,

        #设置权重衰减
        weight_decay=0.01,

        #学习率预热比例
        warmup_ratio=0.01,

        #每轮训练保存模型
        save_strategy="epoch",

        #每轮训练进行评估
        eval_strategy="epoch",

        #自动加载最优模型
        load_best_model_at_end=True,

        #使用F1作为最佳模型指标
        metric_for_best_model="f1",

        #指标越大越优
        greater_is_better=True,

        #设置日志的保存位置
        logging_dir="./logs",

        #设置日志的输出间隔
        logging_steps=10,

        #使用TensorBoard记录
        report_to="tensorboard",

        #Windows关闭多进程加载
        dataloader_num_workers=0,

        #支持GPU启动混合精度
        fp16=torch.cuda.is_available(),

        #设置随机种子
        seed=42,
    )
    print("=================== 6.加载评价指标 ===================")
    accuracy_metric = evaluate.load("accuracy")

    f1_metric = evaluate.load("f1")

    #定义模型评价函数
    def compute_metrics(eval_pred):
        #获取模型输出的真实标签
        logits , label =eval_pred

        #获取最大概率类别
        predictions = np.argmax(
            logits,
            axis=-1
        )

        #计算准确率
        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=label,
        )

        #计算F1值
        f1 = f1_metric.compute(
            predictions=predictions,
            references=label,
            average="binary"
        )

        #返回评价结果
        return {
            "accuracy" : accuracy["accuracy"],
            "f1" : f1["f1"]
        }


    print("=================== 7.创建Trainer训练器 ===================")
    trainer = Trainer(
        #模型信息
        model=model,
        #训练参数
        args=training_args,
        #训练集
        train_dataset=train_dataset,
        #测试集
        eval_dataset=test_dataset,
        #padding
        data_collator=data_collator,
        #评价指标
        compute_metrics=compute_metrics,
    )
    print("=================== 8.开始训练 ===================")
    trainer_result = trainer.train()

    #输出训练损失
    print("训练Loss:",trainer_result.metrics["train_loss"])


    print("=================== 9.模型评估 ===================")
    result = trainer.evaluate()
    #输出准确率
    print("准确率Accuracy：",result["eval_accuracy"])

    #输出f1值
    print("F1:",result["eval_f1"])

    print("=================== 10.模型保存 ===================")
    #设置保存路径
    save_path="./saved_model/imdb-classifier"

    #保存训练后的模型
    trainer.save_model(save_path)

    #保存Tokenizer
    tokenizer.save_pretrained(save_path)

    #压缩模型
    shutil.make_archive(
        "imdb-classifier",
        "zip",
        save_path,
    )

    print("-------------------------- 模型保存完成 ------------------------")

if __name__ == "__main__":
    main()