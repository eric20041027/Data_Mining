# ============================================================
# Phase 3：Full Stack 最終 Ensemble 校準
#
# 合併所有模型：
#   - final_d 原始 4 模型（20 folds）
#   - DeBERTa-v3-large seed42（5 folds）  ← Phase 1
#   - PubMedBERT + pseudo seed42（5 folds）← Phase 2
#
# 執行前提：Phase 1 + Phase 2 均已完成
# 預計時間：~10 分鐘
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, glob, time
import numpy as np
import pandas as pd

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
import tarfile
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
            with tarfile.open(tar_path) as tar:
                tar.extractall(REPO_DIR)
            print(f'✓ Restored from {tar_path}')
            break
else:
    print(f'✓ {already} run dirs present — skip restore')

# 確認 Phase 1 & 2 的模型都在
for pattern in ['deberta_v3_large_seed42_fold*', 'pubmedbert_pseudo_seed42_fold*']:
    found = glob.glob(os.path.join(RUNS_DIR, pattern))
    status = '✅' if len(found) == 5 else f'⚠ {len(found)}/5 folds'
    print(f'  {status}  {pattern}')


# %%
# ========== Cell 3：確認 GT cache ==========
import pickle
CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
if not os.path.exists(CACHE):
    print('hf_gt_cache.pkl missing — uploading...')
    from google.colab import files as colab_files
    uploaded = colab_files.upload()
    for fname, data in uploaded.items():
        with open(CACHE, 'wb') as f:
            f.write(data)
        print(f'✓ Saved to {CACHE}')
else:
    with open(CACHE, 'rb') as f:
        gt = pickle.load(f)
    print(f'✓ GT cache: {len(gt)} labels')


# %%
# ========== Cell 4：Full Stack Ensemble 校準 ==========
CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE) else []

ALL_MODELS = [
    # 原始 final_d 4 模型
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
    # Phase 1: DeBERTa-v3-large
    'outputs/bert_runs/deberta_v3_large_seed42_fold*',
    # Phase 2: PubMedBERT + pseudo
    'outputs/bert_runs/pubmedbert_pseudo_seed42_fold*',
]

cmd = [
    'python', 'src/calibrate.py',
    '--bert-runs', *ALL_MODELS,
    '--no-overlap-constraint',
    '--method', 'both',
    '--tag', 'fullstack',
    *prior_flag,
]

print('Running Full Stack calibration...')
print(f'Models: {len(ALL_MODELS)} groups')
t0 = time.time()
r = subprocess.run(cmd, cwd=REPO_DIR, text=True)
print(f'\n{"✅" if r.returncode==0 else "❌"} 完成 ({(time.time()-t0)/60:.1f} min)')


# %%
# ========== Cell 5：驗證分布 + 估算 LB ==========
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

ref_tags = ['final_d_4noweight', 'cal4_vec_prior']
new_tags  = ['fullstack_raw', 'fullstack_vec', 'fullstack_vec_prior']

print(f'{"Submission":<32}  {"cls1":>5}{"cls2":>5}{"cls3":>5}{"cls4":>5}{"cls5":>5}  {"cls5%":>6}')
print('-'*68)
for tag in ref_tags + new_tags:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index()
    row = ''.join(f'{dist.get(k,0):>5}' for k in range(1, 6))
    marker = ' ★' if tag == 'cal4_vec_prior' else ''
    print(f'{tag:<32}  {row}  {dist.get(5,0)/len(df)*100:>5.1f}%{marker}')

if os.path.exists(CACHE):
    subs = [os.path.join(SUBS_DIR, f'submission_{t}.csv')
            for t in ref_tags + new_tags
            if os.path.exists(os.path.join(SUBS_DIR, f'submission_{t}.csv'))]
    print(f'\nEstimating LB for {len(subs)} submissions...\n')
    result = subprocess.run(
        ['python', 'src/estimate_lb.py'] + subs,
        cwd=REPO_DIR, capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print('STDERR:', result.stderr[:300])


# %%
# ========== Cell 6：下載最佳結果 ==========
from google.colab import files as colab_files
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

priority = [
    'fullstack_vec_prior',   # full stack + vec + prior（預期最強）
    'fullstack_vec',         # full stack + vec
    'fullstack_raw',         # full stack 無校準（對照）
]

downloaded = 0
for tag in priority:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'  submission_{tag}.csv')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ Downloaded {downloaded} files')
print('\n提交判斷：est_lb > 0.65197（cal4_vec_prior）才提交')
