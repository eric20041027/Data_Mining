# ============================================================
# F1-Threshold Optimisation（直接最大化 Macro F1）
#
# 與 calibrate.py 的差異：
#   calibrate.py  → 最小化 NLL（proxy metric）
#   threshold_opt → 直接最大化 OOF Macro F1（比賽真實目標）
#
# 目前最佳：cal4_vec_prior 實際 LB = 0.65197
# 目標：用 F1-threshold 進一步提升
#
# 執行時間：約 3-5 分鐘（Differential Evolution 全域優化）
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

from google.colab import drive
drive.mount('/content/drive')

subprocess.run(['pip', 'install', '-q', 'scipy'], check=True)

# 確認 threshold_opt.py 存在
path = os.path.join(REPO_DIR, 'src', 'threshold_opt.py')
if os.path.exists(path):
    ok = 'differential_evolution' in open(path).read()
    print(f'  {"✓" if ok else "❌"} src/threshold_opt.py')
else:
    print('  ❌ threshold_opt.py missing')

print('Setup OK ✓')


# %%
# ========== Cell 2：還原備份（已有則跳過）==========
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
# ========== Cell 4：Method 1 — 純 F1-threshold（regularise=0.3）==========
# regularise 0.05 → 0.3：防止 OOF overfit（上次 0.05 全部退步）

FINAL_D = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]
CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE) else []

cmd = [
    'python', 'src/threshold_opt.py',
    '--bert-runs', *FINAL_D,
    '--no-overlap-constraint',
    '--tag', 'f1opt4r03',
    '--method', 'f1',
    '--regularise', '0.3',
    *prior_flag,
]

print('Running:', ' '.join(cmd))
print('='*70)
t0 = time.time()
result = subprocess.run(cmd, cwd=REPO_DIR, text=True)
print(f'\n{"✅" if result.returncode==0 else "❌"} 完成 ({(time.time()-t0)/60:.1f} min)')


# %%
# ========== Cell 5：Method 2 — vec → F1（regularise=0.3）==========
# vec scaling 先行 + F1 微調，regularise 加大防 overfit

cmd2 = [
    'python', 'src/threshold_opt.py',
    '--bert-runs', *FINAL_D,
    '--no-overlap-constraint',
    '--tag', 'vf1opt4r03',
    '--method', 'vec+f1',
    '--regularise', '0.3',
    *prior_flag,
]

print('Running:', ' '.join(cmd2))
print('='*70)
t0 = time.time()
result2 = subprocess.run(cmd2, cwd=REPO_DIR, text=True)
print(f'\n{"✅" if result2.returncode==0 else "❌"} 完成 ({(time.time()-t0)/60:.1f} min)')


# %%
# ========== Cell 6：驗證輸出 ==========
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

# Reference: current best
fd_path  = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')
cal_path = os.path.join(SUBS_DIR, 'submission_cal4_vec_prior.csv')

print(f'{"Submission":<35}  {"cls1":>5}{"cls2":>5}{"cls3":>5}{"cls4":>5}{"cls5":>5}  {"cls5%":>6}')
print('-'*70)

refs = [('final_d_4noweight', fd_path), ('cal4_vec_prior [BEST]', cal_path)]
for name, path in refs:
    if os.path.exists(path):
        df = pd.read_csv(path)
        dist = df['label'].value_counts().sort_index()
        row = ''.join(f'{dist.get(k,0):>5}' for k in range(1,6))
        print(f'{name:<35}  {row}  {dist.get(5,0)/len(df)*100:>5.1f}%')

print()
new_tags = [
    'vf1opt4r03_raw', 'vf1opt4r03_vec', 'vf1opt4r03_f1opt', 'vf1opt4r03_f1opt_prior',
    'f1opt4r03_raw',  'f1opt4r03_f1opt', 'f1opt4r03_f1opt_prior',
]

for tag in new_tags:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if not os.path.exists(path): continue
    df = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index()
    row = ''.join(f'{dist.get(k,0):>5}' for k in range(1,6))
    print(f'{tag:<35}  {row}  {dist.get(5,0)/len(df)*100:>5.1f}%')


# %%
# ========== Cell 7：估算 LB ==========
CACHE = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
if not os.path.exists(CACHE):
    print('⚠️  hf_gt_cache.pkl missing')
else:
    new_subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_f1opt4r03*.csv')))
    new_subs += sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_vf1opt4r03*.csv')))
    if os.path.exists(cal_path): new_subs.append(cal_path)
    if os.path.exists(fd_path):  new_subs.append(fd_path)

    print(f'Estimating LB for {len(new_subs)} submissions...\n')
    subprocess.run(['python', 'src/estimate_lb.py'] + new_subs, cwd=REPO_DIR)


# %%
# ========== Cell 8：下載 ==========
from google.colab import files as colab_files
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

# 優先下載 F1-opt 結果
priority = [
    'vf1opt4r03_f1opt_prior',  # vec + F1-opt(r=0.3) + prior
    'vf1opt4r03_f1opt',        # vec + F1-opt(r=0.3)
    'f1opt4r03_f1opt_prior',   # 純 F1-opt(r=0.3) + prior
    'f1opt4r03_f1opt',         # 純 F1-opt(r=0.3)
    'vf1opt4r03_vec_prior',    # vec + prior（對照，應與 cal4_vec_prior 相同）
]

downloaded = 0
for tag in priority:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'  submission_{tag}.csv')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ Downloaded {downloaded} files')
print('\n提交優先順序：依 Cell 7 的 est_lb 由高到低')
print('重要：OOF F1 Δ > 0 才提交（可能 overfit OOF，需謹慎）')
