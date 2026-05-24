# ============================================================
# Kaggle 醫學文本分類 — Ensemble Aggregation 實驗腳本
# 目的：用幾何平均 / Power Mean 取代算術平均，嘗試突破 0.64596
# 把這個檔案的每個 cell 依序貼進 Colab 執行
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess

REPO_DIR = '/content/Data_Mining'

# clone 或更新 repo（確保包含最新的 ensemble_agg.py）
if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
os.environ['HF_HOME'] = '/content/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

# 掛 Drive
from google.colab import drive
drive.mount('/content/drive')

# 確認 ensemble_agg.py 存在
assert os.path.exists('src/ensemble_agg.py'), \
    "❌ src/ensemble_agg.py 不存在！請先把本地改動 git push 到 GitHub 再 pull。"

# 安裝套件（如果已裝過可跳過）
subprocess.run(['pip', 'install', '-q', '-U',
                'transformers>=4.44,<4.50', 'accelerate>=0.33',
                'datasets>=2.20', 'scikit-learn>=1.4'], check=True)

import torch
print('CUDA :', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Setup OK ✓')


# %%
# ========== Cell 2：從 Drive 還原 prediction artifacts ==========
import tarfile, glob

# 自動找最新備份（從最優先到備用）
BACKUP_CANDIDATES = [
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_after_noweight.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_final_all_noweight.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_before_noweight.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only.tar.gz',
]

restored = False
for tar_path in BACKUP_CANDIDATES:
    if os.path.exists(tar_path):
        print(f'Restoring from: {tar_path}')
        with tarfile.open(tar_path) as tar:
            tar.extractall(REPO_DIR)
        print('  done ✓')
        restored = True
        break

if not restored:
    print('⚠️  找不到備份！請手動上傳 tar.gz 到 Drive 或重新訓練。')

# 列出還原到的 run 資料夾
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
if os.path.isdir(RUNS_DIR):
    dirs = sorted(d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d)))
    print(f'\n=== 共 {len(dirs)} 個 run 資料夾 ===')
    for d in dirs:
        rd = os.path.join(RUNS_DIR, d)
        has_test = os.path.exists(os.path.join(rd, 'test_probs.npy'))
        has_val  = os.path.exists(os.path.join(rd, 'val_probs.npy'))
        print(f'  {"✓" if has_test and has_val else "⚠"} {d}  (test_probs={has_test}, val_probs={has_val})')
else:
    print('❌ outputs/bert_runs 不存在')


# %%
# ========== Cell 3：確認 final_d 4 個模型都在 ==========
import json

FINAL_D_PATTERNS = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

print('=== final_d 模型完整性檢查 ===')
all_ok = True
for pattern in FINAL_D_PATTERNS:
    matches = sorted(glob.glob(pattern))
    has_all_folds = len(matches) == 5
    for_each = all(
        os.path.exists(os.path.join(d, 'test_probs.npy')) and
        os.path.exists(os.path.join(d, 'val_probs.npy'))
        for d in matches
    )
    status = '✓' if has_all_folds and for_each else '❌'
    all_ok = all_ok and has_all_folds and for_each
    print(f'  {status} {pattern}  ({len(matches)} folds)')

if all_ok:
    print('\n✅ 所有 final_d 模型完整，可以開始跑 ensemble_agg')
else:
    print('\n⚠️  部分模型缺失。缺失的需要重訓或從 Drive 補充。')


# %%
# ========== Cell 4：跑四種 Aggregation，輸出 4 份 submission ==========
# arith  = 算術平均（等同現在的 final_d，用作基準確認）
# geom   = 幾何平均（log 空間平均，對邊界案例更保守）
# pow2   = Power Mean p=2（放大高信心預測）
# pow05  = Power Mean p=0.5（軟化高信心預測）

cmd = [
    'python', 'src/ensemble_agg.py',
    '--bert-runs',
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
    '--no-overlap-constraint',
    '--tag', 'agg',
]

print('Running:', ' '.join(cmd))
print('='*60)
result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode != 0:
    print(f'\n❌ ensemble_agg.py 執行失敗 (returncode={result.returncode})')
else:
    print('\n✅ ensemble_agg.py 執行完成')

# 驗證輸出
import pandas as pd
print('\n=== 輸出 submission 驗證 ===')
for method in ['arith', 'geom', 'pow2', 'pow05']:
    path = f'outputs/submissions/submission_agg_{method}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        dist = df['label'].value_counts().sort_index().to_dict()
        print(f'  ✓ {method}: {dist}')
    else:
        print(f'  ❌ {method}: 檔案不存在')


# %%
# ========== Cell 5：與 final_d baseline 比較 ==========
import numpy as np

methods = ['arith', 'geom', 'pow2', 'pow05']

# 讀 final_d baseline (已在 outputs/submissions/)
baseline_path = 'outputs/submissions/submission_final_d_4noweight.csv'
if os.path.exists(baseline_path):
    fd = pd.read_csv(baseline_path)
    print(f'final_d baseline loaded: {len(fd)} rows')
    print(f'final_d dist: {fd["label"].value_counts().sort_index().to_dict()}')
    print()

    print('=== 各方法 vs final_d 差異 ===')
    print(f'{"Method":<8}  {"Changed":>8}  {"Class 1":>7}  {"Class 2":>7}  {"Class 3":>7}  {"Class 4":>7}  {"Class 5":>7}')
    print('-'*65)
    for method in methods:
        path = f'outputs/submissions/submission_agg_{method}.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        changed = (df['label'] != fd['label']).sum()
        dist = df['label'].value_counts().sort_index()
        print(f'{method:<8}  {changed:>8}  ' + '  '.join(f'{dist.get(k, 0):>7}' for k in range(1, 6)))
else:
    print('⚠️  final_d baseline 不存在，跳過比較')


# %%
# ========== Cell 6：下載所有 submission ==========
from google.colab import files

to_download = []

# 4 個 aggregation 方法
for method in ['arith', 'geom', 'pow2', 'pow05']:
    path = f'/content/Data_Mining/outputs/submissions/submission_agg_{method}.csv'
    if os.path.exists(path):
        to_download.append(path)
    else:
        print(f'⚠️  找不到: {path}')

# 也順便下載 final_d baseline 確認格式
baseline = '/content/Data_Mining/outputs/submissions/submission_final_d_4noweight.csv'
if os.path.exists(baseline):
    to_download.append(baseline)

print(f'\n下載 {len(to_download)} 份 submission:')
for p in to_download:
    print(f'  {os.path.basename(p)}')
    files.download(p)

print('\n✅ 下載完成。提交順序建議：geom → pow05 → pow2 → arith（arith 應等同 final_d）')


# %%
# ========== Cell 7（選用）：備份新的 predictions 到 Drive ==========
# 如果本次 Colab session 有重新訓練模型，記得備份
# 如果只是跑 ensemble_agg（沒有重訓），可以跳過

def backup(label):
    src = '/content/Data_Mining/outputs/bert_runs'
    dst = f'/content/drive/MyDrive/Kaggle_backup/predictions_only_{label}.tar.gz'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    files_to_pack = []
    for run in sorted(os.listdir(src)):
        rd = os.path.join(src, run)
        if not os.path.isdir(rd):
            continue
        for pat in ['*.npy', '*.json', '*.csv']:
            files_to_pack += glob.glob(os.path.join(rd, pat))
    with tarfile.open(dst, 'w:gz') as tar:
        for f in files_to_pack:
            tar.add(f, arcname=os.path.relpath(f, '/content/Data_Mining'))
    size = subprocess.check_output(['du', '-h', dst]).decode().split()[0]
    print(f'Backup saved: {dst} ({size})')

# 取消下方註解再執行（只有重新訓練才需要）
# backup('after_ensemble_agg')
