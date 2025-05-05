#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


df = pd.read_csv('/kaggle/input/train1/train1.csv', header=0, low_memory=False)

df.head()

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Convert to float for multi-label classification in pytorch
df["labels"] = df[label_cols].values.tolist()
df["labels"] = df[label_cols].apply(lambda row: row.astype(float).tolist(), axis=1)


train_df, test_df = train_test_split(df[["comment_text", "labels"]], test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})


df.head()

from transformers import AutoTokenizer

# Used for Kaggle since it can't connect to the internet to get BERT model
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/bert-base-uncased/bert-base-uncased")


# Use this otherwise
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["comment_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)


# Used to check if correct BERT files were imported
#import os
#os.listdir("/kaggle/input/bert-base-uncased")
#os.listdir("/kaggle/input/bert-base-uncased/bert-base-uncased")


dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# If not using Kaggle, replace directory with bert-base-uncased
model = BertForSequenceClassification.from_pretrained(
    "/kaggle/input/bert-classification",
    num_labels=6
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()  # Ensure CPU before .numpy()
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}


# In[32]:


training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


import os
# Disable W&B from hugging face
os.environ["WANDB_DISABLED"] = "true"

trainer.train()

def predict_toxicity(text, model, tokenizer, label_cols, threshold=0.5):
    model.eval()

    # Automatically detect the model's device (cuda or cpu)
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)  # Move input tensors to same device as model

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    
    preds = (probs > threshold).astype(int)
    label_map = dict(zip(label_cols, preds))

    print(f"\nInput: {text}\nPredicted categories:")
    for label, value in label_map.items():
        if value == 1:
            print(f" - {label}")
    if sum(preds) == 0:
        print(" - (none)")

    return label_map

predict_toxicity("test comment", model, tokenizer, label_cols)

