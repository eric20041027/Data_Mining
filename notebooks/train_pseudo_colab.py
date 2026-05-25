# ============================================================
# Phase 2：Pseudo-labeling（用當前最佳 ensemble 對 test 做偽標籤再訓練）
#
# 合法性：偽標籤來自我們自己的模型預測，非任何外部 GT
# 預計時間：~60 分鐘（pseudo 生成 5 min + 5 folds × ~10 min）
# 執行前提：Phase 1（DeBERTa）已完成，bert_runs 有 20+ 目錄
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, tarfile, json, time
import numpy as np

REPO_DIR = '/content/Data_Mining'

if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone',
        'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
print('Setup OK ✓')


# %%
# ========== Cell 2：還原備份 ==========
from google.colab import drive
drive.mount('/content/drive')

RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
already = sum(1 for d in os.listdir(RUNS_DIR)
              if os.path.isdir(os.path.join(RUNS_DIR, d))) if os.path.isdir(RUNS_DIR) else 0

if already < 20:
    for tar_path in [
        '/content/drive/MyDrive/Kaggle_backup/predictions_FINAL_0524.tar.gz',
        '/content/drive/MyDrive/Kaggle_backup/predictions_only_after_noweight.tar.gz',
    ]:
        if os.path.exists(tar_path):
            print(f'Restoring: {tar_path}')
            with tarfile.open(tar_path) as tar:
                tar.extractall(REPO_DIR)
            print('done ✓')
            break
else:
    print(f'✓ {already} run dirs already present — skip restore')


# %%
# ========== Cell 3：生成 Pseudo-labels ==========
# 使用 final_d 4 模型 ensemble（confidence >= 0.80）

PSEUDO_PATTERNS = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]
PSEUDO_CSV = 'outputs/pseudo_labels.csv'

cmd_pseudo = [
    'python', 'src/make_pseudo_labels.py',
    '--bert-runs', *PSEUDO_PATTERNS,
    '--threshold', '0.80',
    '--out', PSEUDO_CSV,
]
print('Generating pseudo-labels...')
r = subprocess.run(cmd_pseudo, cwd=REPO_DIR, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print('❌ STDERR:', r.stderr[:500])
    raise RuntimeError('make_pseudo_labels failed')
print('✅ Pseudo-labels ready')


# %%
# ========== Cell 4：訓練 PubMedBERT + pseudo（5 folds）==========
MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
TAG   = 'pubmedbert_pseudo_seed42'

fold_f1s = []
t0_total = time.time()

for fold in range(5):
    print(f'\n{"="*60}')
    print(f'  PubMedBERT+Pseudo  Fold {fold}/4')
    print(f'{"="*60}', flush=True)
    t0 = time.time()

    cmd_str = (
        f'cd {REPO_DIR} && python src/train_bert_pseudo.py'
        f' --model {MODEL}'
        f' --fold {fold}'
        f' --seed 42'
        f' --epochs 4'
        f' --batch-size 32'
        f' --lr 2e-5'
        f' --max-length 512'
        f' --pseudo-csv {PSEUDO_CSV}'
        f' --class-weight balanced'
        f' --tag {TAG}_fold{fold}'
    )
    ret = os.system(cmd_str)
    elapsed = (time.time() - t0) / 60
    status = '✅' if ret == 0 else f'❌ (exit {ret})'
    print(f'\n{status} Fold {fold} done ({elapsed:.1f} min)', flush=True)
    if ret != 0:
        print('⚠ 訓練失敗')
        break

    metrics_path = os.path.join(REPO_DIR, 'outputs', 'bert_runs',
                                f'{TAG}_fold{fold}', 'metrics.json')
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))
        f1 = m.get('val_macro_f1', 0)
        fold_f1s.append(f1)
        print(f'  Val Macro F1: {f1:.4f}')

total_min = (time.time() - t0_total) / 60
print(f'\n{"="*60}')
print(f'Pseudo 訓練完成（{total_min:.1f} min）')
if fold_f1s:
    avg = np.mean(fold_f1s)
    print(f'  Per-fold F1: {[f"{f:.4f}" for f in fold_f1s]}')
    print(f'  Avg val F1:  {avg:.4f}')
    print(f'  Δ vs PubMedBERT noweight (0.6524): {avg - 0.6524:+.4f}')
