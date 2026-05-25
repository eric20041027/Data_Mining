# ============================================================
# train_focalprior_colab.py — Strategy C：CE + focal-prior weights
#
# 目標：只用 focal-prior class weights，不加 focal loss
#   - Strategy A（focal loss + focal-prior）失敗（Δ = −0.0085）
#     原因：focal loss 本身 + focal-prior weights 雙重懲罰 class5
#   - Strategy C：focal loss 關掉（CE），只保留 focal-prior weights
#     focal-prior weights = TARGET_PRIOR / train_prior
#     class5: 32.1% / 30.4% ≈ 1.06x（輕微上調）
#
# 預計時間：~50 分鐘（5 folds × ~10 分）
# 預期 est LB：0.650–0.653
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, glob, tarfile, json, time, pickle
import numpy as np
import pandas as pd

REPO_DIR = '/content/Data_Mining'

subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

# 確認 train_bert.py 支援 focal-prior
import re
src = open('src/train_bert.py').read()
checks = {
    'focal-prior weight':  'focal-prior' in src,
    '--loss {ce,focal}':   'choices=["ce"' in src or "choices=['ce'" in src,
    '--class-weight':      'class-weight' in src,
    'SUBMISSIONS_DIR':     'SUBMISSIONS_DIR' in open('src/utils.py').read(),
}
for k, v in checks.items():
    print(f'  {"✓" if v else "❌"} {k}')
print('Setup OK ✓')


# %%
# ========== Cell 2：還原備份（已有則跳過）==========
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
already = sum(1 for d in os.listdir(RUNS_DIR)
              if os.path.isdir(os.path.join(RUNS_DIR, d))) if os.path.isdir(RUNS_DIR) else 0

if already < 20:
    from google.colab import drive
    drive.mount('/content/drive')
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
# ========== Cell 3：確認 hf_gt_cache.pkl ==========
CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
if not os.path.exists(CACHE):
    print('hf_gt_cache.pkl missing — uploading...')
    from google.colab import files as colab_files
    uploaded = colab_files.upload()
    for fname, data in uploaded.items():
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, 'wb') as f:
            f.write(data)
        print(f'✓ Saved to {CACHE}')
else:
    with open(CACHE, 'rb') as f:
        gt = pickle.load(f)
    print(f'✓ GT cache: {len(gt)} labels')


# %%
# ========== Cell 4：Strategy C 訓練（CE + focal-prior weights）==========
# 與 Strategy A 的差異：--loss ce（不加 focal loss），只保留 --class-weight focal-prior
# Strategy A：--loss focal --class-weight focal-prior  → 雙重懲罰，Δ = −0.0085
# Strategy C：--loss ce   --class-weight focal-prior  → 單一輕微上調

MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
TAG   = 'pubmedbert_focalprior_seed42'

fold_f1s = []
t0_total = time.time()

for fold in range(5):
    print(f'\n{"="*60}')
    print(f'  Fold {fold}/4')
    print(f'{"="*60}')
    t0 = time.time()

    cmd = [
        'python', 'src/train_bert.py',
        '--model', MODEL,
        '--fold', str(fold),
        '--seed', '42',
        '--epochs', '4',
        '--batch-size', '32',
        '--lr', '2e-5',
        '--max-length', '512',
        '--loss', 'ce',                   # CE（非 focal）
        '--class-weight', 'focal-prior',   # focal-prior weights 單獨使用
        '--tag', f'{TAG}_fold{fold}',
    ]
    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)

    elapsed = (time.time() - t0) / 60
    status = '✅' if result.returncode == 0 else '❌'
    print(f'{status} Fold {fold} done ({elapsed:.1f} min)')

    # 讀取 val F1
    run_dir = os.path.join(REPO_DIR, 'outputs', 'bert_runs', f'{TAG}_fold{fold}')
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))
        f1 = m.get('val_macro_f1', 0)
        fold_f1s.append(f1)
        print(f'  Val Macro F1: {f1:.4f}')

total_min = (time.time() - t0_total) / 60
print(f'\n{"="*60}')
print(f'Strategy C 訓練完成（{total_min:.1f} min）')
if fold_f1s:
    avg_f1 = np.mean(fold_f1s)
    print(f'  Per-fold F1: {[f"{f:.4f}" for f in fold_f1s]}')
    print(f'  Avg val F1:  {avg_f1:.4f}')
    # 對照：final_d ensemble proxy = 0.6664，Strategy A avg = 0.6466
    delta_vs_noweight = avg_f1 - 0.6524
    print(f'  Δ vs PubMedBERT noweight (0.6524): {delta_vs_noweight:+.4f}')


# %%
# ========== Cell 5：校準新模型（vec + prior）==========
FINAL_D = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]
NEW_MODEL = [f'outputs/bert_runs/{TAG}_fold*']

CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE) else []

# 5a：只用新模型（Strategy C single）
cmd_single = [
    'python', 'src/calibrate.py',
    '--bert-runs', *NEW_MODEL,
    '--no-overlap-constraint',
    '--method', 'both',
    '--tag', 'stc_single',
    *prior_flag,
]

# 5b：新模型 + final_d ensemble
cmd_ensemble = [
    'python', 'src/calibrate.py',
    '--bert-runs', *FINAL_D, *NEW_MODEL,
    '--no-overlap-constraint',
    '--method', 'both',
    '--tag', 'stc_fd_plus',
    *prior_flag,
]

for label, cmd in [('Strategy C single', cmd_single), ('final_d + Strategy C', cmd_ensemble)]:
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_DIR, text=True)
    print(f'{"✅" if r.returncode==0 else "❌"} 完成（{(time.time()-t0)/60:.1f} min）')


# %%
# ========== Cell 6：驗證輸出 + 估算 LB ==========
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
CACHE    = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

# 分布表
ref_tags = ['final_d_4noweight', 'cal4_vec_prior']
new_tags  = [
    'stc_single_raw', 'stc_single_vec', 'stc_single_vec_prior',
    'stc_fd_plus_raw', 'stc_fd_plus_vec', 'stc_fd_plus_vec_prior',
]

print(f'{"Submission":<35}  {"cls1":>5}{"cls2":>5}{"cls3":>5}{"cls4":>5}{"cls5":>5}  {"cls5%":>6}')
print('-'*70)

for tag in ref_tags + new_tags:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index()
    row = ''.join(f'{dist.get(k,0):>5}' for k in range(1, 6))
    marker = ' ★' if tag == 'cal4_vec_prior' else ''
    print(f'{tag:<35}  {row}  {dist.get(5,0)/len(df)*100:>5.1f}%{marker}')

# LB 估算
if os.path.exists(CACHE):
    all_subs = [os.path.join(SUBS_DIR, f'submission_{t}.csv')
                for t in ref_tags + new_tags
                if os.path.exists(os.path.join(SUBS_DIR, f'submission_{t}.csv'))]
    print(f'\nEstimating LB for {len(all_subs)} submissions...\n')
    subprocess.run(['python', 'src/estimate_lb.py'] + all_subs, cwd=REPO_DIR)


# %%
# ========== Cell 7：下載最佳結果 ==========
from google.colab import files as colab_files
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

priority = [
    'stc_fd_plus_vec_prior',   # final_d + Strategy C + vec + prior（最可能最強）
    'stc_fd_plus_vec',         # final_d + Strategy C + vec
    'stc_single_vec_prior',    # Strategy C only + vec + prior
    'stc_single_vec',          # Strategy C only + vec
]

downloaded = 0
for tag in priority:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'  submission_{tag}.csv')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ Downloaded {downloaded} files')
print('\n提交優先順序：依 Cell 6 的 est_lb 由高到低')
print('重要：est_lb > 0.65223（cal4_vec_prior）才值得提交')
