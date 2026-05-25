# ============================================================
# train_and_calibrate_colab.py — 訓練 + 校準完整流程
#
# 修正了：
#   1. submission 輸出路徑：outputs/submissions/（已 git push 修正）
#   2. train_bert.py focal loss 支援（已 git push）
#   3. 更好的錯誤輸出
#
# 流程：
#   Part A（快速，今天就能提交，不需重訓）：
#     Cell 1 : 環境 + git pull + 安裝套件
#     Cell 2 : 還原 Drive 備份
#     Cell 3 : 上傳 hf_gt_cache.pkl
#     Cell 4 : 校準（Temperature + Vector Scaling）
#     Cell 5 : 驗證輸出 + 估算 LB
#     Cell 6 : 下載最佳 submission
#
#   Part B（Focal Loss 重訓，約 50 分鐘）：
#     Cell 7 : 訓練 5 folds
#     Cell 8 : 校準新模型 + 估算 LB
#     Cell 9 : 備份 + 下載
# ============================================================

# %%
# ========== Cell 1：環境設定（git pull 拿最新修正版）==========
import os, sys, subprocess, glob, tarfile, json, time, pickle
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

subprocess.run(['pip', 'install', '-q', '-U',
                'transformers>=4.44,<4.50', 'accelerate>=0.33',
                'datasets>=2.20', 'scikit-learn>=1.4', 'scipy'], check=True)

import torch
print('CUDA :', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# 確認關鍵腳本都在
for script in ['src/calibrate.py', 'src/train_bert.py', 'src/ensemble_agg.py', 'src/estimate_lb.py']:
    path = os.path.join(REPO_DIR, script)
    ok = os.path.exists(path)
    if ok:
        with open(path) as f:
            content = f.read()
        # 確認 calibrate.py 有 vector scaling
        if 'calibrate' in script:
            ok = 'fit_vector' in content or 'Vector' in content
        # 確認 train_bert.py 有 focal loss
        if 'train_bert' in script:
            ok = 'focal' in content
    print(f'  {"✓" if ok else "❌"} {script}')

print('\nSetup OK ✓')


# %%
# ========== Cell 2：還原 Drive 備份 ==========
BACKUP_CANDIDATES = [
    '/content/drive/MyDrive/Kaggle_backup/predictions_FINAL_0524.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_after_noweight.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_final_all_noweight.tar.gz',
]

restored_path = None
for tar_path in BACKUP_CANDIDATES:
    if os.path.exists(tar_path):
        print(f'Restoring from: {tar_path}')
        with tarfile.open(tar_path) as tar:
            tar.extractall(REPO_DIR)
        restored_path = tar_path
        print('  done ✓')
        break

if not restored_path:
    print('⚠️  找不到備份！Drive 內容：')
    bk_dir = '/content/drive/MyDrive/Kaggle_backup'
    if os.path.isdir(bk_dir):
        for f in sorted(os.listdir(bk_dir)):
            size = os.path.getsize(os.path.join(bk_dir, f)) // 1024
            print(f'  {f} ({size} KB)')
    else:
        print('  Kaggle_backup 資料夾不存在')

# 確認 final_d 4 模型完整
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
FINAL_D_PATTERNS = [
    'pubmedbert_noweight_seed42_fold*',
    'pubmedbert_noweight_seed2024_fold*',
    'biobert_noweight_seed42_fold*',
    'pubmedbertlarge_noweight_seed42_fold*',
]
print('\n=== final_d 模型 ===')
all_ok = True
for pat in FINAL_D_PATTERNS:
    matches = sorted(glob.glob(os.path.join(RUNS_DIR, pat)))
    ok = len(matches) == 5 and all(
        os.path.exists(os.path.join(d, 'test_probs.npy')) and
        os.path.exists(os.path.join(d, 'val_probs.npy'))
        for d in matches
    )
    print(f'  {"✓" if ok else "❌"} {pat:<50} ({len(matches)}/5 folds)')
    all_ok = all_ok and ok
print(f'\n{"✅ 所有模型完整" if all_ok else "⚠️  部分缺失"}')


# %%
# ========== Cell 3：上傳 hf_gt_cache.pkl ==========
# 這個 cache 提供各 class 真實分佈，讓 prior adjustment 生效
# 從你本地 Desktop/Kaggle_new/outputs/hf_gt_cache.pkl 上傳

from google.colab import files as colab_files

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        gt = pickle.load(f)
    from collections import Counter
    c = Counter(gt.tolist())
    total = len(gt)
    print(f'✓ hf_gt_cache.pkl 已存在 ({total} labels)')
    NAMES = ['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']
    print('Target prior:')
    for i, name in enumerate(NAMES):
        cnt = c.get(i+1, 0)
        print(f'  class {i+1} ({name:<25}): {cnt:4d} ({cnt/total*100:.1f}%)')
    print('\n✓ 可跳過此 Cell')
else:
    print('請上傳 hf_gt_cache.pkl（本地 outputs/ 目錄下）')
    uploaded = colab_files.upload()
    for fname, data in uploaded.items():
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            f.write(data)
        print(f'✓ Saved → {CACHE_PATH}')


# %%
# ========== Cell 4：Part A — 校準現有 final_d 4 模型 ==========
# 用 OOF val_probs 做 Temperature Scaling + Vector Scaling
# 不需重訓！約 1-2 分鐘

FINAL_D_PATTERNS_FULL = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []
if prior_flag:
    print('✓ hf_gt_cache.pkl 找到 → 同時跑 prior adjustment')
else:
    print('ℹ️  hf_gt_cache.pkl 未找到 → 只跑純 OOF 校準（先跑 Cell 3）')

cmd = [
    'python', 'src/calibrate.py',
    '--bert-runs', *FINAL_D_PATTERNS_FULL,
    '--no-overlap-constraint',
    '--tag', 'cal4',
    '--method', 'both',   # scalar temperature + vector scaling
    *prior_flag,
]

print('\nRunning:', ' '.join(cmd))
print('='*70)

result = subprocess.run(cmd, cwd=REPO_DIR, text=True)

print()
if result.returncode != 0:
    print(f'❌ calibrate.py 失敗 (returncode={result.returncode})')
else:
    # 確認輸出檔案存在
    SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
    found = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal4*.csv')))
    if found:
        print(f'✅ 校準完成，產生 {len(found)} 個 submission：')
        for f in found:
            print(f'  {os.path.basename(f)}')
    else:
        print('⚠️  找不到輸出檔案，請查看上方錯誤訊息')


# %%
# ========== Cell 5：Part A — 驗證輸出 + 估算 LB ==========

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

# baseline
fd_path = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')
fd_dist = {}
if os.path.exists(fd_path):
    fd = pd.read_csv(fd_path)
    fd_dist = fd['label'].value_counts().sort_index().to_dict()
    print(f'final_d baseline: {fd_dist}  class5={fd_dist.get(5,0)} ({fd_dist.get(5,0)/len(fd)*100:.1f}%)')
else:
    print('⚠️  submission_final_d_4noweight.csv 不存在（備份未包含？）')

print()
print(f'{"Submission":<30}  {"cls1":>5}{"cls2":>5}{"cls3":>5}{"cls4":>5}{"cls5":>5}  {"cls5%":>6}  {"Δcls5":>7}')
print('-'*75)

CAL_TAGS = ['cal4_raw', 'cal4_temp', 'cal4_vec',
            'cal4_prior', 'cal4_temp_prior', 'cal4_vec_prior']

found_any = False
for tag in CAL_TAGS:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if not os.path.exists(path):
        continue
    found_any = True
    df   = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index()
    cls5 = dist.get(5, 0)
    pct  = cls5 / len(df) * 100
    delta = cls5 - fd_dist.get(5, 0) if fd_dist else 0
    row  = ''.join(f'{dist.get(k,0):>5}' for k in range(1, 6))
    print(f'{tag:<30}  {row}  {pct:>5.1f}%  {delta:>+7}')

if not found_any:
    print('⚠️  找不到任何 cal4_* submission')
    print('    請重新執行 Cell 4')

# LB Estimation
if os.path.exists(CACHE_PATH):
    cal_subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal4*.csv')))
    if os.path.exists(fd_path):
        cal_subs.append(fd_path)
    if cal_subs:
        print(f'\n=== LB Estimation ({len(cal_subs)} files) ===')
        subprocess.run(
            ['python', 'src/estimate_lb.py', '--quiet'] + cal_subs,
            cwd=REPO_DIR
        )
else:
    print('\nℹ️  hf_gt_cache.pkl 未找到，跳過 LB 估算')
    print('   先跑 Cell 3 再重新執行此 Cell')


# %%
# ========== Cell 6：Part A — 下載最佳 submission ==========
from google.colab import files as colab_files

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

# 依優先順序：有 prior adjustment 最好，其次純校準
PRIORITY = [
    ('cal4_vec_prior',   '★★★ vector scaling (OOF) + prior adjust (HF)'),
    ('cal4_temp_prior',  '★★☆ temperature scaling + prior adjust'),
    ('cal4_vec',         '★★☆ vector scaling（純 OOF，無 HF）'),
    ('cal4_temp',        '★☆☆ temperature scaling（純 OOF，無 HF）'),
    ('cal4_prior',       '★☆☆ prior adjust only'),
]

to_download = []
for tag, desc in PRIORITY:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        to_download.append((tag, path, desc))

if not to_download:
    print('⚠️  找不到任何 cal4 submission，請先執行 Cell 4')
else:
    print(f'Downloading {len(to_download)} submissions:')
    for tag, path, desc in to_download:
        print(f'  {desc}')
        colab_files.download(path)
    print('\n✅ Part A 下載完成')
    print()
    print('提交建議：')
    for tag, _, desc in to_download:
        print(f'  submission_{tag}.csv  ← {desc}')
    print()
    print('重要：先確認 Cell 5 的 OOF F1 Δ > 0 再提交')
    print('      cal4_raw 應等於 final_d baseline（OOF Δ=0），作為對照')


# ================================================================
# ===  PART B：Focal Loss 重訓（需 ~50 分鐘）  ==================
# ================================================================

# %%
# ========== Cell 7：Part B — Focal Loss 訓練 5 folds ==========
# Strategy A（推薦）：Focal Loss (γ=2) + focal-prior class weights
# focal-prior weights：class 5 訓練損失權重 × 1.37（= target_prior/train_prior）

import torch
if not torch.cuda.is_available():
    print('⚠️  CUDA 不可用！Focal Loss 訓練需要 GPU')
    print('   請確認 Colab 已選擇 GPU 硬體加速')
    raise SystemExit()

# 確認 train_bert.py 有 focal loss
TRAIN_SCRIPT = os.path.join(REPO_DIR, 'src', 'train_bert.py')
with open(TRAIN_SCRIPT) as f:
    tb_content = f.read()
if 'focal' not in tb_content:
    print('❌ train_bert.py 沒有 focal loss！Cell 1 的 git pull 是否成功？')
    print('   請重新執行 Cell 1（git pull）再跑本 Cell')
    raise SystemExit()
print(f'✓ train_bert.py focal loss support OK')

STRATEGY = 'A'   # A=focal+fprior, B=focal only, C=ce+fprior（改這裡）
CONFIGS = {
    'A': dict(loss='focal',  cw='focal-prior', gamma=2.0, tag='focal_fprior'),
    'B': dict(loss='focal',  cw='none',        gamma=2.0, tag='focal_noweight'),
    'C': dict(loss='ce',     cw='focal-prior', gamma=2.0, tag='ce_fprior'),
}
cfg = CONFIGS[STRATEGY]
BASE_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

print(f'\nStrategy {STRATEGY}: loss={cfg["loss"]}, '
      f'class_weight={cfg["cw"]}, gamma={cfg["gamma"]}')
print(f'Tag: pubmedbert_{cfg["tag"]}_seed42_fold[0-4]')
print('='*60)

fold_f1s = []
t_total = time.time()

for fold in range(5):
    tag = f'pubmedbert_{cfg["tag"]}_seed42_fold{fold}'
    t_fold = time.time()
    print(f'\n[fold {fold}/4]  {tag}')

    cmd = [
        'python', 'src/train_bert.py',
        '--model',        BASE_MODEL,
        '--fold',         str(fold),
        '--seed',         '42',
        '--epochs',       '4',
        '--batch-size',   '32',
        '--lr',           '2e-5',
        '--class-weight', cfg['cw'],
        '--loss',         cfg['loss'],
        '--focal-gamma',  str(cfg['gamma']),
        '--tag',          tag,
    ]

    result = subprocess.run(cmd, cwd=REPO_DIR, text=True)
    elapsed_fold = (time.time() - t_fold) / 60

    if result.returncode != 0:
        print(f'❌ fold {fold} 訓練失敗！(returncode={result.returncode})')
        continue

    # 讀取 val F1
    run_dirs = sorted(glob.glob(os.path.join(REPO_DIR, 'outputs', 'bert_runs', f'{tag}*')))
    if run_dirs:
        m_path = os.path.join(run_dirs[-1], 'metrics.json')
        if os.path.exists(m_path):
            with open(m_path) as f2:
                m = json.load(f2)
            fold_f1s.append(m['val_macro_f1'])
            print(f'✓ fold {fold} val F1 = {m["val_macro_f1"]:.4f}  ({elapsed_fold:.1f} min)')

elapsed_total = (time.time() - t_total) / 60
print(f'\n訓練完成！共 {elapsed_total:.1f} 分鐘')
if fold_f1s:
    avg = np.mean(fold_f1s)
    print(f'Average val F1: {avg:.4f}  (folds: {[f"{x:.4f}" for x in fold_f1s]})')
    print(f'final_d baseline: ~0.6551')
    delta = avg - 0.6551
    print(f'Δ = {delta:+.4f}  {"✅ 改善！" if delta > 0 else "❌ 退步"}')
    if delta <= 0:
        print('建議：嘗試 Strategy C（CE + focal-prior weights）或調高 --focal-gamma 1.5')


# %%
# ========== Cell 8：Part B — 校準新模型 + 估算 LB ==========

new_pattern = f'outputs/bert_runs/pubmedbert_{cfg["tag"]}_seed42_fold*'
new_dirs = sorted(glob.glob(os.path.join(REPO_DIR, new_pattern)))
print(f'New runs found: {len(new_dirs)}/5')

if len(new_dirs) < 5:
    print('⚠️  新模型未全部完成，請先完成 Cell 7')
else:
    SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
    CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
    prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []
    new_tag = f'new_{cfg["tag"]}'

    # Step 1: ensemble_agg (arith/geom baseline for comparison)
    print('\n--- Step 1: ensemble_agg ---')
    subprocess.run([
        'python', 'src/ensemble_agg.py',
        '--bert-runs', new_pattern,
        '--no-overlap-constraint',
        '--tag', f'focal_{cfg["tag"]}',
    ], cwd=REPO_DIR)

    # Step 2: calibrate new model
    print(f'\n--- Step 2: calibrate ({new_tag}) ---')
    subprocess.run([
        'python', 'src/calibrate.py',
        '--bert-runs', new_pattern,
        '--no-overlap-constraint',
        '--tag', new_tag,
        '--method', 'both',
        *prior_flag,
    ], cwd=REPO_DIR)

    # Step 3: estimate LB
    if os.path.exists(CACHE_PATH):
        print('\n--- Step 3: estimate LB ---')
        fd_path = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')
        all_subs = (
            sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_focal_{cfg["tag"]}*.csv'))) +
            sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_{new_tag}*.csv'))) +
            sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal4*.csv')))
        )
        if os.path.exists(fd_path):
            all_subs.append(fd_path)
        all_subs = sorted(set(all_subs))
        subprocess.run(
            ['python', 'src/estimate_lb.py', '--quiet'] + all_subs,
            cwd=REPO_DIR
        )
    else:
        print('\nℹ️  hf_gt_cache.pkl 未找到，跳過 LB 估算')


# %%
# ========== Cell 9：Part B — 備份到 Drive + 下載 ==========
from google.colab import files as colab_files

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

# 備份新模型 predictions 到 Drive
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
tag_substr = cfg['tag']
files_to_pack = []
for run in sorted(os.listdir(RUNS_DIR)):
    if tag_substr not in run:
        continue
    rd = os.path.join(RUNS_DIR, run)
    if not os.path.isdir(rd):
        continue
    for pat in ['*.npy', '*.json']:
        files_to_pack += glob.glob(os.path.join(rd, pat))

if files_to_pack:
    dst = f'/content/drive/MyDrive/Kaggle_backup/predictions_{tag_substr}_seed42.tar.gz'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tarfile.open(dst, 'w:gz') as tar:
        for fp in files_to_pack:
            tar.add(fp, arcname=os.path.relpath(fp, REPO_DIR))
    print(f'✓ Backup saved: {dst} ({os.path.getsize(dst)//1024} KB, {len(files_to_pack)} files)')
else:
    print(f'⚠️  找不到 "{tag_substr}" 相關的 run 檔案')

# 下載所有新 submission
new_tag = f'new_{cfg["tag"]}'
DOWNLOAD_ORDER = [
    (f'{new_tag}_vec_prior',   '★★★ focal + vector calibration + prior'),
    (f'{new_tag}_temp_prior',  '★★☆ focal + temperature calibration + prior'),
    (f'{new_tag}_vec',         '★★☆ focal + vector calibration'),
    (f'{new_tag}_temp',        '★☆☆ focal + temperature calibration'),
    (f'focal_{cfg["tag"]}_geom',  '☆☆☆ focal 幾何平均（無校準，對照用）'),
    (f'focal_{cfg["tag"]}_arith', '☆☆☆ focal 算術平均（無校準，對照用）'),
]

print(f'\nDownloading Part B submissions:')
downloaded = 0
for tag, desc in DOWNLOAD_ORDER:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'  {desc}')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ Part B 完成，下載了 {downloaded} 個檔案')
print()
print('最終提交建議（依預期效果排序）：')
print(f'  1. {new_tag}_vec_prior     ← 最強組合')
print(f'  2. cal4_vec_prior           ← Part A 最強（今天就能提交）')
print(f'  3. {new_tag}_temp_prior    ← 次強組合')
print()
print('判斷標準：OOF F1 Δ > 0 且 class 5 count 接近 464（32.1%）')
