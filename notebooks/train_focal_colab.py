# ============================================================
# Focal Loss + Focal-Prior 重訓練腳本
# 目的：從訓練層面修正 class 5 under-prediction
#
# 核心修改：
#   1. --loss focal --focal-gamma 2.0
#      → 對預測錯的 hard example 給更大梯度（class 5 通常最 hard）
#   2. --class-weight focal-prior
#      → 訓練損失 class 5 權重 ∝ target_prior / train_prior ≈ 1.37x
#   3. 保持其他超參不變，直接和 final_d 4模型比較 OOF F1
#
# 預計改善：
#   - Class 5 recall: 42.2% → 55%+
#   - OOF Macro F1: 0.6551 → 0.67+
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, glob, json, time
import numpy as np
import pandas as pd

REPO_DIR = '/content/Data_Mining'

if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
os.environ['HF_HOME'] = '/content/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

from google.colab import drive
drive.mount('/content/drive')

import torch
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Setup OK ✓')


# %%
# ========== Cell 2：安裝套件 ==========
subprocess.run(['pip', 'install', '-q', '-U',
                'transformers>=4.44,<4.50', 'accelerate>=0.33',
                'datasets>=2.20', 'scikit-learn>=1.4', 'scipy'], check=True)
print('Packages OK ✓')


# %%
# ========== Cell 3：訓練設定 ==========
# 選擇訓練策略
# Strategy A: focal loss + focal-prior weights（推薦，最針對 class 5）
# Strategy B: focal loss only（不改 weights）
# Strategy C: focal-prior weights only（不改 loss）

STRATEGY = 'A'   # ← 改這裡

BASE_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

STRATEGY_CONFIGS = {
    'A': {
        'loss': 'focal',
        'class_weight': 'focal-prior',
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'tag_suffix': 'focal_fprior',
        'desc': 'Focal Loss (γ=2) + focal-prior weights',
    },
    'B': {
        'loss': 'focal',
        'class_weight': 'none',
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'tag_suffix': 'focal_noweight',
        'desc': 'Focal Loss (γ=2), no class weights',
    },
    'C': {
        'loss': 'ce',
        'class_weight': 'focal-prior',
        'focal_gamma': 2.0,
        'label_smoothing': 0.0,
        'tag_suffix': 'ce_fprior',
        'desc': 'Cross-Entropy + focal-prior weights',
    },
}

cfg = STRATEGY_CONFIGS[STRATEGY]
print(f'Strategy {STRATEGY}: {cfg["desc"]}')
print(f'  loss={cfg["loss"]}, class_weight={cfg["class_weight"]}, '
      f'focal_gamma={cfg["focal_gamma"]}, label_smoothing={cfg["label_smoothing"]}')


# %%
# ========== Cell 4：訓練所有 5 個 fold（PubMedBERT base）==========
# 預計時間：每個 fold ≈ 8-12 分鐘，共 5 fold ≈ 50-60 分鐘

FOLDS = range(5)
SEEDS = [42]          # 可改成 [42, 2024] 多 seed

all_fold_results = []
t_start = time.time()

for seed in SEEDS:
    for fold in FOLDS:
        tag = f'pubmedbert_{cfg["tag_suffix"]}_seed{seed}_fold{fold}'
        print(f'\n{"="*60}')
        print(f'Training: {tag}')
        print(f'{"="*60}')

        cmd = [
            'python', 'src/train_bert.py',
            '--model', BASE_MODEL,
            '--fold', str(fold),
            '--seed', str(seed),
            '--epochs', '4',
            '--batch-size', '32',
            '--lr', '2e-5',
            '--class-weight', cfg['class_weight'],
            '--loss', cfg['loss'],
            '--focal-gamma', str(cfg['focal_gamma']),
            '--label-smoothing', str(cfg['label_smoothing']),
            '--tag', tag,
        ]

        result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)
        if result.returncode != 0:
            print(f'❌ fold {fold} 訓練失敗！')
        else:
            # Read val F1 from metrics.json
            run_dirs = sorted(glob.glob(
                os.path.join(REPO_DIR, 'outputs', 'bert_runs', f'{tag}*')
            ))
            if run_dirs:
                metrics_path = os.path.join(run_dirs[-1], 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        m = json.load(f)
                    all_fold_results.append(m)
                    print(f'✓ fold {fold} val Macro F1 = {m["val_macro_f1"]:.4f}')

elapsed = time.time() - t_start
print(f'\n\nTotal training time: {elapsed/60:.1f} min')

if all_fold_results:
    avg_f1 = np.mean([r['val_macro_f1'] for r in all_fold_results])
    print(f'Average val Macro F1: {avg_f1:.4f}')


# %%
# ========== Cell 5：跑 ensemble_agg 比較新舊模型 ==========
import subprocess

# New focal model patterns
focal_pattern = f'outputs/bert_runs/pubmedbert_{cfg["tag_suffix"]}_seed42_fold*'

# Baseline final_d patterns
fd_patterns = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

print('=== Running ensemble for new focal model ===')
cmd_new = [
    'python', 'src/ensemble_agg.py',
    '--bert-runs', focal_pattern,
    '--no-overlap-constraint',
    '--tag', f'new_{cfg["tag_suffix"]}',
]
subprocess.run(cmd_new, cwd=REPO_DIR)

print('\n=== Running ensemble for final_d baseline ===')
cmd_fd = [
    'python', 'src/ensemble_agg.py',
    '--bert-runs', *fd_patterns,
    '--no-overlap-constraint',
    '--tag', 'fd4_check',
]
subprocess.run(cmd_fd, cwd=REPO_DIR)


# %%
# ========== Cell 6：跑 calibrate.py 加校準 ==========
# 在新模型上疊加溫度校準，進一步提升

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []

cmd_cal = [
    'python', 'src/calibrate.py',
    '--bert-runs', focal_pattern,
    '--no-overlap-constraint',
    '--tag', f'cal_{cfg["tag_suffix"]}',
    '--method', 'both',
    *prior_flag,
]
print('Running calibration on new focal model...')
subprocess.run(cmd_cal, cwd=REPO_DIR)


# %%
# ========== Cell 7：估算 LB ==========
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

if not os.path.exists(CACHE_PATH):
    print('⚠️  hf_gt_cache.pkl not found — skipping LB estimate')
    print('   Upload it and rerun this cell')
else:
    # Collect all new submissions
    new_subs = sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_new_{cfg["tag_suffix"]}*.csv')))
    new_subs += sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_cal_{cfg["tag_suffix"]}*.csv')))
    # Add baseline for comparison
    fd_path = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')
    if os.path.exists(fd_path):
        new_subs.append(fd_path)

    if new_subs:
        print(f'Estimating LB for {len(new_subs)} submissions...')
        result = subprocess.run(
            ['python', 'src/estimate_lb.py', '--quiet'] + new_subs,
            cwd=REPO_DIR
        )


# %%
# ========== Cell 8：備份新模型 ==========
import tarfile, os

def backup_new_runs(label):
    src = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
    dst = f'/content/drive/MyDrive/Kaggle_backup/predictions_{label}.tar.gz'
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    tag_substr = cfg['tag_suffix']
    files_to_pack = []
    for run in sorted(os.listdir(src)):
        if tag_substr not in run:
            continue
        rd = os.path.join(src, run)
        if not os.path.isdir(rd):
            continue
        for pat in ['*.npy', '*.json']:
            files_to_pack += glob.glob(os.path.join(rd, pat))

    if not files_to_pack:
        print('No files to pack — check tag_suffix')
        return

    with tarfile.open(dst, 'w:gz') as tar:
        for f in files_to_pack:
            tar.add(f, arcname=os.path.relpath(f, REPO_DIR))

    size_kb = os.path.getsize(dst) // 1024
    print(f'Backup saved: {dst} ({size_kb} KB, {len(files_to_pack)} files)')

backup_new_runs(f'focal_{cfg["tag_suffix"]}_seed42')


# %%
# ========== Cell 9：下載 submission ==========
from google.colab import files as colab_files

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

to_download = []
# Calibrated focal submissions (best)
for suffix in ['vec_prior', 'temp_prior', 'vec', 'temp']:
    p = os.path.join(SUBS_DIR, f'submission_cal_{cfg["tag_suffix"]}_{suffix}.csv')
    if os.path.exists(p):
        to_download.append(p)

# Raw ensemble (arith/geom)
for suffix in ['arith', 'geom']:
    p = os.path.join(SUBS_DIR, f'submission_new_{cfg["tag_suffix"]}_{suffix}.csv')
    if os.path.exists(p):
        to_download.append(p)

print(f'Downloading {len(to_download)} submissions:')
for p in to_download:
    print(f'  {os.path.basename(p)}')
    colab_files.download(p)

print('\n✅ 完成！')
print()
print('提交建議（先確認 OOF F1 和 LB 估算再決定）：')
print(f'  1. cal_{cfg["tag_suffix"]}_vec_prior   ← focal + vector calibration + prior')
print(f'  2. cal_{cfg["tag_suffix"]}_temp_prior  ← focal + temperature calibration + prior')
print(f'  3. new_{cfg["tag_suffix"]}_geom        ← focal 純幾何平均（對照組）')
