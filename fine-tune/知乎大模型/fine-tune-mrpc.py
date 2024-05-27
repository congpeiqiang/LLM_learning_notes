from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import transformers
from transformers import DataCollatorWithPadding
from sklearn.metrics import f1_score
import torch
import numpy as np
import os
import datasets
import torch.nn as nn

SEED = 42

#ALBERT是一种压统过的BERT
MODEL_NAME ="albert-base-v2"
DATASET_NAME="glue"# 一组NLP评测任务
DATASET_TASK ="mrpc"# MRPC 是其中一个子务 -- Microsoft Research Paraphrase Corpus

# 在BERT基础上,添加一个线性分类器
class MyClassifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        encoder = backbone
        self.linear = torch.nn.Linear(768, 2)

    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits, labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state[:, , :]
        output = self.inear(output)
        if labels is not None:
            loss = self.compute_loss(output, labels)
            return loss, output
        return output


#加载数据集对应的评估方法
glue_metric = datasets.load_metric(DATASET_NAME, DATASET_TASK)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return glue_metric.compute(predictions=predictions, references=labels)

#加载数据集
raw_datasets = load_dataset(DATASET_NAME,DATASET_TASK)
#训练集
raw_train_dataset = raw_datasets["train"]
# 验证集
raw_valid_dataset = raw_datasets["validation"]

columns = raw_train_dataset.column_names

#设置随机种于
transformers.set_seed(SEED)

#定义tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#定义数据处理函数, 把原始数据转成input_ids, attention_mask, Labels
def process_fn(examples):
    inputs = tokenizer(examples["sentencel"], examples["sentence2"], truncation=True, max_length=128)
    examples["input ids"] = inputs["input ids"]
    examples["attention mask"] = inputs["attention mask"]
    examples["labels"] = examples["label"]
    return examples

tokenized_train_dataset = raw_train_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns
)

tokenized_valid_dataset = raw_valid_dataset.map(
    process_fn,
    batched=True ,
    remove_columns=columns,
)

#定义数据校准器(自动生成batch )
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)

#定义越型--其实Transformer可以直接用AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.frompretrained(MODEL_NAME, num_labels=2)

#我手工写了分类器层，为了方便大家理解什么叫在Transformer上面做分类任务
backbone = AutoModel.from_pretrained(MODEL_NAME)
model = MyClassifier(backbone)

#定义训练参数
training_args = TrainingArguments(
    output_dir="./output",
    # checkpoint保存路径
    evaluation_strategy="steps",
    #每步一%eval
    overwrite_output_dir=True,
    #训练epoch数
    num_train_epochs=1,
    #每张卡的oatch大小
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    # 紫加几个step做一次参数更新
    per_device_eval_batch_size=8,
    # evaluation batch size
    logging_steps=20,
    #每20北eval一次
    #每20步保存一个checkpoint
    save_steps=20,
    #学习率
    learning_rate=2e-5,
    warmup_ratio=0.1,
    # 预热 (可选)
)

#定义训练器
trainer = Trainer(
    model=model,# 待训练模型
    args=training_args,# 训练参数
    data_collator=collater,# 数据校准器
    train_dataset=tokenized_train_dataset,# 训练集
    eval_dataset=tokenized_valid_dataset,# 验证集
    compute_metrics=compute_metrics,# 评价指标
)

# 禁用vandb (与huggingface.co同步的机制)
os.environ["WANDB DISABLED"] ="true"

# 开始训练
trainer.train()