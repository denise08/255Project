# -*- coding: utf-8 -*-
"""TF-IDF+SVM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hvBBaif-ezk4f9kkEVSq0S3XeUJlD3T-
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    hamming_loss
)

df = pd.read_csv('/content/train1.csv')
df.dropna(subset=['comment_text'], inplace=True)

def clean_text(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

df['clean'] = df['comment_text'].map(clean_text)

LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
X = df['clean'].values
y = df[LABELS].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

base_svc = LinearSVC(dual=False, C=1.0, max_iter=2000)
calibrated = CalibratedClassifierCV(base_svc, method='sigmoid', cv=3)
clf = OneVsRestClassifier(calibrated, n_jobs=-1)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=5)),
    ('clf',   clf)
])

# Model : SVM for Multi-Label
print("Training SVM-based toxicity classifier...")
pipeline.fit(X_train, y_train)

from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score
)
y_pred  = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)
acc = accuracy_score(y_val, y_pred)
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=LABELS, zero_division=0, digits=4))
print(f"Accuracy (exact match):   {acc:.4f}")
print("Micro F1: ", f1_score(y_val, y_pred, average='micro'))
print("Macro F1: ", f1_score(y_val, y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(y_val, y_pred))

roc_aucs = {}
for i, lbl in enumerate(LABELS):
    if len(np.unique(y_val[:,i]))>1:
        roc_aucs[lbl] = roc_auc_score(y_val[:,i], y_proba[:,i])
    else:
        roc_aucs[lbl] = float('nan')

print("\nROC AUC by label:")
for lbl, auc in roc_aucs.items():
    print(f"  {lbl:15s}: {auc:.4f}")
print("Mean ROC AUC:", np.nanmean(list(roc_aucs.values())))

print("\n=== Toxicity Checker (SVM + TF–IDF) ===")
print("Type a comment and press Enter. Type 'exit' to quit.\n")

while True:
    txt = input("Your comment: ").strip()
    if txt.lower() in ('exit','quit'):
        break
    txt_cl = clean_text(txt)
    probs = pipeline.predict_proba([txt_cl])[0]
    preds = (probs >= 0.5).astype(int)

    print("\nPredictions:")
    for lbl, p, pr in zip(LABELS, probs, preds):
        bar = '█' * int(p*20)
        print(f"  {lbl:15s} | {p:.3f} | {bar} | pred={pr}")
    print("\n" + "-"*50 + "\n")

