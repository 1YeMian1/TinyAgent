# ============================================================
# Transformer文本分类
# DistilBERT实现IMDB情感分析
# 流程: 数据 -> Tokenizer -> 模型 -> 训练 -> 保存
# ============================================================
# 设置HuggingFace国内镜像
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
import torch
import numpy as np

# 加载数据集
from datasets import load_dataset

# 导入Transformer组件
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# 加载评价指标
import evaluate


# 主程序入口
def main():

    # ==============================
    # 1.加载数据集
    # ==============================

    print("1.准备数据集")

    # 加载IMDB数据集
    dataset = load_dataset("stanfordnlp/imdb")

    # 获取训练集
    train_dataset = dataset["train"]

    # 获取测试集
    test_dataset = dataset["test"]

    print(dataset.shape)
    print(train_dataset[0])


    # ==============================
    # 2.Tokenizer文本编码
    # ==============================

    print("2.Tokenizer转换")

    # 指定预训练模型
    model_name = "distilbert/distilbert-base-uncased"

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # 定义文本转换方法
    def tokenize(example):

        # 文本转Token
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding=False
        )


    # 训练集编码
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    # 测试集编码
    test_dataset = test_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )


    # ==============================
    # 3.Padding数据对齐
    # ==============================

    print("3.Padding处理")

    # 动态补齐长度
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )


    # ==============================
    # 4.加载Transformer模型
    # ==============================

    print("4.加载模型")

    # 加载文本分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={
            0: "NEGATIVE",
            1: "POSITIVE"
        },
        label2id={
            "NEGATIVE": 0,
            "POSITIVE": 1
        }
    )


    # ==============================
    # 5.训练参数配置
    # ==============================

    print("5.训练配置")

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./output",

        # 训练轮数
        num_train_epochs=5,

        # Batch大小
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        # 学习率
        learning_rate=2e-5,

        # 权重衰减
        weight_decay=0.01,

        # 学习率预热
        warmup_ratio=0.01,

        # 保存策略
        save_strategy="epoch",

        # 评估策略
        eval_strategy="epoch",

        # 加载最佳模型
        load_best_model_at_end=True,

        # 最佳指标
        metric_for_best_model="f1",

        # 日志
        logging_dir="./logs",
        logging_steps=10,

        # TensorBoard
        report_to="tensorboard",

        # Windows关闭多进程
        dataloader_num_workers=0,

        # GPU混合精度
        fp16=torch.cuda.is_available(),

        # 随机种子
        seed=42
    )


    # ==============================
    # 6.评价指标
    # ==============================

    print("6.加载指标")

    # 加载准确率
    accuracy_metric = evaluate.load("accuracy")

    # 加载F1
    f1_metric = evaluate.load("f1")


    # 定义评价函数
    def compute_metrics(eval_pred):

        # 获取预测结果
        logits, labels = eval_pred

        # 获取最大概率类别
        predictions = np.argmax(
            logits,
            axis=-1
        )

        # 计算准确率
        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels
        )

        # 计算F1
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="binary"
        )

        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"]
        }


    # ==============================
    # 7.Trainer训练模型
    # ==============================

    print("7.创建Trainer")


    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    # 开始训练
    print("开始训练")

    train_result = trainer.train()


    # 输出训练结果
    print(
        "训练Loss:",
        train_result.metrics["train_loss"]
    )


    # ==============================
    # 8.模型评估
    # ==============================

    print("8.模型评估")


    # 测试模型
    result = trainer.evaluate()


    print(
        "Accuracy:",
        result["eval_accuracy"]
    )

    print(
        "F1:",
        result["eval_f1"]
    )


    # ==============================
    # 9.保存模型
    # ==============================

    print("9.保存模型")


    # 模型保存路径
    save_path = "./saved_model/imdb-classifier"


    # 保存模型
    trainer.save_model(save_path)

    # 保存Tokenizer
    tokenizer.save_pretrained(save_path)


    # ==============================
    # 10.压缩模型
    # ==============================

    print("10.压缩模型")


    # 生成zip文件
    shutil.make_archive(
        "imdb-classifier",
        "zip",
        save_path
    )


    print("模型保存完成")


# Windows入口保护
if __name__ == "__main__":
    main()