# BERT Classification + Clustering + WordCloud + External Test Set Evaluation

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os

# --- 1. Load Training Data ---
train_df = pd.read_csv('/kaggle/input/train1/train1.csv', header=0, low_memory=False)
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_df["labels"] = train_df[label_cols].apply(lambda row: row.astype(float).tolist(), axis=1)

# --- 2. Train-Test Split ---
train_data, val_data = train_test_split(train_df[["comment_text", "labels"]], test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_data),
    "test": Dataset.from_pandas(val_data)
})

# --- 3. Tokenizer ---
tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert-base-uncased/bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["comment_text"], padding="max_length", truncation=True, max_length=128)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- 4. Load Model ---
model = BertForSequenceClassification.from_pretrained("/kaggle/input/bert-classification", num_labels=6)

# --- 5. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

# --- 6. Trainer Setup ---
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
    compute_metrics=compute_metrics
)

# --- 7. Train Model ---
os.environ["WANDB_DISABLED"] = "true"
trainer.train()
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

dataset["train"].reset_format()
dataset["train"].set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "comment_text"]
)


# --- 8. Extract CLS Embeddings ---
def extract_cls_embeddings(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=16)
    embeddings, labels, texts = [], [], []

    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embed)
            labels.extend(batch['labels'].cpu().numpy())
        texts.extend(batch['comment_text'])

    return np.vstack(embeddings), np.array(labels), texts

cls_embeddings, label_array, comment_texts = extract_cls_embeddings(model, dataset["train"])

# --- 9. SVD + Clustering ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cls_embeddings)
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X_reduced)

# --- 10. Plot Clusters ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=cluster_labels, palette='viridis')
plt.title("KMeans Clustering of BERT CLS Embeddings (2D SVD)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# --- 11. WordClouds per Cluster ---
comment_array = np.array(comment_texts)
for i in range(5):
    cluster_comments = comment_array[cluster_labels == i]
    text = " ".join(cluster_comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud for Cluster {i}")
    plt.show()

# --- 12. Load External Test Set ---
external_test_df = pd.read_csv("/kaggle/input/test-data/test.csv")  # expects 'comment_text' column
test_dataset = Dataset.from_pandas(external_test_df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# --- 13. Predict on Test Set ---
def predict_batch(dataset, model, threshold=0.5):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=16)
    results = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            results.extend(preds)
    return results

test_predictions = predict_batch(test_dataset, model)
pred_df = pd.DataFrame(test_predictions, columns=label_cols)
final_output = pd.concat([external_test_df, pred_df], axis=1)
final_output.to_csv("final_predictions.csv", index=False)


def predict_comment(text, model, tokenizer, label_cols, threshold=0.5):
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


if __name__ == "__main__":

    print("=== Toxic Comment Classifier ===")
    print("Type a comment and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Your comment: ").strip()
        if text.lower() in ('exit', 'quit'):
            print("Goodbye, Have A Great Day!")
            break


        predict_comment(text, model, tokenizer, label_cols)
