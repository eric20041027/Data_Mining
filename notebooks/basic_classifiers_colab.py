# ============================================================
# Table 1：基本分類器實驗（補齊報告所需缺失數據）
#
# 目的：
#   - NB / LR / DT / KNN / SVM 各 3 組參數，共 15 個實驗
#   - 每個實驗：5-fold CV 取 Accuracy + Macro P/R/F1
#   - 每個實驗：生成 submission CSV，上傳 Kaggle 取 Public LB
#
# 執行環境：Google Colab（CPU 即可，不需要 GPU）
# 預計時間：~15 分鐘（KNN 最慢）
# ============================================================

# %% Cell 1：環境設定
import os, sys, subprocess

REPO_DIR = '/content/Data_Mining'
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone',
        'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'], check=True)
os.chdir(REPO_DIR)
print('Setup OK ✓')

# %% Cell 2：載入資料
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

df_train = pd.read_csv(os.path.join(REPO_DIR, 'kaggle_trainset.csv'))
df_test  = pd.read_csv(os.path.join(REPO_DIR, 'kaggle_testset.csv'))

X_train = df_train['condition'].fillna('').astype(str).values
X_test  = df_test['condition'].fillna('').astype(str).values

# 固定標籤順序（對應競賽 1-5）
LABEL_ORDER = [
    'neoplasms',                    # 1
    'digestive system diseases',    # 2
    'nervous system diseases',      # 3
    'cardiovascular diseases',      # 4
    'general pathological conditions',  # 5
]
label2num = {c: i+1 for i, c in enumerate(LABEL_ORDER)}

le = LabelEncoder()
le.fit(LABEL_ORDER)
y_train = le.transform(df_train['label'].values)

print(f'Train: {len(X_train)}  Test: {len(X_test)}')
print(f'Classes: {LABEL_ORDER}')

# %% Cell 3：定義所有分類器（15 組）
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TFIDF = dict(ngram_range=(1,2), min_df=3, max_df=0.95,
             sublinear_tf=True, max_features=80_000)

configs = [
    # Naïve Bayes
    ('NB #1',  'MultinomialNB(alpha=1.0)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)), ('c', MultinomialNB(alpha=1.0))])),
    ('NB #2',  'MultinomialNB(alpha=0.5)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)), ('c', MultinomialNB(alpha=0.5))])),
    ('NB #3',  'MultinomialNB(alpha=0.1)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)), ('c', MultinomialNB(alpha=0.1))])),
    # Logistic Regression
    ('LR #1',  'LogisticRegression(C=1.0)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])),
    ('LR #2',  'LogisticRegression(C=5.0)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LogisticRegression(C=5.0, max_iter=1000, random_state=42))])),
    ('LR #3',  'LogisticRegression(C=0.1, class_weight=balanced)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LogisticRegression(C=0.1, max_iter=1000,
                                        class_weight='balanced', random_state=42))])),
    # Decision Tree
    ('DT #1',  'DecisionTreeClassifier(max_depth=20)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', DecisionTreeClassifier(max_depth=20, random_state=42))])),
    ('DT #2',  'DecisionTreeClassifier(max_depth=50)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', DecisionTreeClassifier(max_depth=50, random_state=42))])),
    ('DT #3',  'DecisionTreeClassifier(min_samples_leaf=5)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', DecisionTreeClassifier(max_depth=None,
                                            min_samples_leaf=5, random_state=42))])),
    # KNN
    ('KNN #1', 'KNeighborsClassifier(n_neighbors=5, metric=cosine)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', KNeighborsClassifier(n_neighbors=5, metric='cosine'))])),
    ('KNN #2', 'KNeighborsClassifier(n_neighbors=11, metric=cosine)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', KNeighborsClassifier(n_neighbors=11, metric='cosine'))])),
    ('KNN #3', 'KNeighborsClassifier(n_neighbors=3, metric=cosine)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', KNeighborsClassifier(n_neighbors=3, metric='cosine'))])),
    # SVM
    ('SVM #1', 'LinearSVC(C=1.0)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LinearSVC(C=1.0, max_iter=3000, random_state=42))])),
    ('SVM #2', 'LinearSVC(C=0.1)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LinearSVC(C=0.1, max_iter=3000, random_state=42))])),
    ('SVM #3', 'LinearSVC(C=5.0)',
     Pipeline([('v', TfidfVectorizer(**TFIDF)),
               ('c', LinearSVC(C=5.0, max_iter=3000, random_state=42))])),
]

# %% Cell 4：執行實驗 + 生成 submission
OUT_DIR = '/content/basic_submissions'
os.makedirs(OUT_DIR, exist_ok=True)

results = []
print(f"{'Method':<10} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print('─' * 50)

for name, params_str, pipe in configs:
    import time; t0 = time.time()

    # 5-fold CV
    oof = np.zeros(len(y_train), dtype=int)
    for tr, va in skf.split(X_train, y_train):
        pipe.fit(X_train[tr], y_train[tr])
        oof[va] = pipe.predict(X_train[va])

    acc  = accuracy_score(y_train, oof)
    f1   = f1_score(y_train, oof, average='macro')
    prec = precision_score(y_train, oof, average='macro')
    rec  = recall_score(y_train, oof, average='macro')

    # Train on full train → predict test
    pipe.fit(X_train, y_train)
    test_preds  = pipe.predict(X_test)
    test_labels = le.inverse_transform(test_preds)
    test_nums   = [label2num[l] for l in test_labels]

    # Save submission
    tag      = name.replace(' ', '').replace('#', '').lower()
    sub_path = os.path.join(OUT_DIR, f'submission_{tag}.csv')
    pd.DataFrame({'id': range(len(test_nums)), 'label': test_nums}).to_csv(
        sub_path, index=False)

    elapsed = time.time() - t0
    results.append({'method': name, 'params': params_str,
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                    'file': f'submission_{tag}.csv'})
    print(f'{name:<10} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}  ({elapsed:.0f}s)')

# %% Cell 5：印出完整結果（回報給 Claude 用）
print('\n' + '=' * 70)
print('【請複製以下結果回報給 Claude】')
print('=' * 70)
for r in results:
    print(f"{r['method']:<10}  acc={r['acc']:.4f}  "
          f"P/R/F1={r['prec']:.4f}/{r['rec']:.4f}/{r['f1']:.4f}  "
          f"file={r['file']}")

# %% Cell 6：下載所有 submission CSV
from google.colab import files as colab_files
import glob

print('\n下載所有 submission CSV...')
for fpath in sorted(glob.glob(os.path.join(OUT_DIR, '*.csv'))):
    fname = os.path.basename(fpath)
    print(f'  ↓ {fname}')
    colab_files.download(fpath)

print('\n✅ 完成！接下來請：')
print('1. 把上方【回報結果】區塊完整複製給 Claude')
print('2. 逐一上傳 CSV 到 Kaggle，把每個的 Public LB Score 和 Rank 也回報給 Claude')
print('3. （Private LB 競賽結束後才會揭曉，可填 — ）')
