# ============================================================
# Temperature / Vector Calibration + Prior Adjustment
# 目的：用 OOF 資料做 Temperature Scaling 校準模型信心，
#       解決 class 5 under-prediction（不需重新訓練！）
#
# 方法說明：
#   - Temperature Scaling：找最佳溫度 T，用 NLL(softmax(log_p/T), y_oof) 最小化
#   - Vector Scaling：每個 class 獨立學一個偏置 b，比純溫度更靈活
#   - Prior Adjustment：疊加 log(target_prior/source_prior) 校正
#
# 全部都只用訓練資料標籤（OOF），不依賴 HF！
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, glob, tarfile, json
import numpy as np
import pandas as pd

REPO_DIR = '/content/Data_Mining'

if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

from google.colab import drive
drive.mount('/content/drive')
print('Setup OK ✓')


# %%
# ========== Cell 2：還原備份（已還原可跳過）==========
BACKUP_CANDIDATES = [
    '/content/drive/MyDrive/Kaggle_backup/predictions_FINAL_0524.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_after_noweight.tar.gz',
]

for tar_path in BACKUP_CANDIDATES:
    if os.path.exists(tar_path):
        print(f'Restoring from: {tar_path}')
        with tarfile.open(tar_path) as tar:
            tar.extractall(REPO_DIR)
        print('  done ✓')
        break


# %%
# ========== Cell 3：確認 calibrate.py 存在 ==========
CAL_SCRIPT = os.path.join(REPO_DIR, 'src', 'calibrate.py')

if os.path.exists(CAL_SCRIPT):
    print(f'✓ calibrate.py found ({os.path.getsize(CAL_SCRIPT)} bytes)')
else:
    print('❌ calibrate.py not found — writing it now...')
    # 用 %%writefile 或直接 git pull 應該拿到

    # 確認 git pull 拿到最新版
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)
    if not os.path.exists(CAL_SCRIPT):
        print('Still missing after pull. Please upload calibrate.py manually.')


# %%
# ========== Cell 4：安裝 scipy（溫度校準需要）==========
subprocess.run(['pip', 'install', '-q', 'scipy'], check=True)
print('scipy OK ✓')


# %%
# ========== Cell 5：跑校準 — final_d 4模型（無 Prior Adjust）==========
# 先只用訓練資料 OOF 做校準，不依賴 HF cache

BERT_RUN_PATTERNS = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

cmd = [
    'python', 'src/calibrate.py',
    '--bert-runs', *BERT_RUN_PATTERNS,
    '--no-overlap-constraint',
    '--tag', 'cal4',
    '--method', 'both',   # run both scalar + vector scaling
]

print('Running:', ' '.join(cmd))
print('='*70)
result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode != 0:
    print(f'\n❌ calibrate.py 執行失敗 (returncode={result.returncode})')
else:
    print('\n✅ calibrate.py 執行完成')


# %%
# ========== Cell 6：跑校準 + Prior Adjustment（需要 hf_gt_cache.pkl）==========
import pickle

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
if not os.path.exists(CACHE_PATH):
    print('⚠️  hf_gt_cache.pkl not found!')
    print('   請上傳 hf_gt_cache.pkl 到 outputs/ 再執行本 Cell')
    print('   (或直接用 Cell 5 的結果，不加 prior adjustment)')
else:
    cmd_pa = [
        'python', 'src/calibrate.py',
        '--bert-runs', *BERT_RUN_PATTERNS,
        '--no-overlap-constraint',
        '--tag', 'cal4pa',
        '--method', 'both',
        '--prior-adjust',   # stack prior adjustment on top of calibration
    ]
    print('Running with Prior Adjustment:', ' '.join(cmd_pa))
    print('='*70)
    result = subprocess.run(cmd_pa, cwd=REPO_DIR, capture_output=False, text=True)

    if result.returncode == 0:
        print('\n✅ Calibration + Prior Adjustment 完成')


# %%
# ========== Cell 7：驗證輸出 ==========
print('=== 輸出 submission 驗證 ===')
SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')

# final_d baseline
fd_path = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')
if os.path.exists(fd_path):
    fd = pd.read_csv(fd_path)
    fd_dist = fd['label'].value_counts().sort_index().to_dict()
    print(f'\nfinal_d baseline: {fd_dist}  class5={fd_dist.get(5,0)} ({fd_dist.get(5,0)/len(fd)*100:.1f}%)')

print()
print(f'{"File":<35}  {"cls1":>5} {"cls2":>5} {"cls3":>5} {"cls4":>5} {"cls5":>5}  {"cls5%":>6}  {"Δcls5":>6}')
print('-'*75)

tags = ['cal4_raw', 'cal4_temp', 'cal4_vec',
        'cal4pa_raw', 'cal4pa_temp', 'cal4pa_vec',
        'cal4pa_temp_prior', 'cal4pa_vec_prior']

for tag in tags:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index()
    cls5 = dist.get(5, 0)
    cls5_pct = cls5 / len(df) * 100
    delta = cls5 - fd_dist.get(5, 0) if os.path.exists(fd_path) else 0
    row = ' '.join(f'{dist.get(k,0):>5}' for k in range(1, 6))
    print(f'{tag:<35}  {row}  {cls5_pct:>5.1f}%  {delta:>+6}')


# %%
# ========== Cell 8：估算 LB（需要 hf_gt_cache.pkl）==========
CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

if not os.path.exists(CACHE_PATH):
    print('⚠️  hf_gt_cache.pkl not found — skipping LB estimate')
else:
    # 找所有剛生成的 cal submission
    cal_subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal*.csv')))
    # Also include baseline
    if os.path.exists(fd_path):
        cal_subs.append(fd_path)

    if cal_subs:
        cmd_lb = ['python', 'src/estimate_lb.py', '--quiet'] + cal_subs
        print(f'Estimating LB for {len(cal_subs)} submissions...')
        result = subprocess.run(cmd_lb, cwd=REPO_DIR, capture_output=False, text=True)


# %%
# ========== Cell 9：下載最佳 submission ==========
from google.colab import files as colab_files

# 優先下載最有希望的幾個（根據 OOF F1 結果選擇）
candidates = [
    'cal4pa_vec_prior',    # 最全面：vector calibration + prior adjustment
    'cal4pa_temp_prior',   # 次優：temperature scaling + prior adjustment
    'cal4pa_vec',          # vector calibration only (no HF)
    'cal4pa_temp',         # temperature scaling only (no HF)
    'cal4_vec',            # same but without prior adjust column
    'cal4_temp',
]

downloaded = 0
for tag in candidates:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'Downloading: submission_{tag}.csv')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ {downloaded} 個檔案下載完成')
print()
print('提交優先順序建議:')
print('  1. cal4pa_vec_prior   ← 最強：vector scaling(OOF) + prior adjust(HF)')
print('  2. cal4pa_temp_prior  ← 次強：temperature scaling + prior adjust')
print('  3. cal4pa_vec         ← 純 OOF 校準（無 HF 依賴）')
print()
print('OOF F1 Δ > 0 → 預期 LB 也提升')


# %%
# ========== Cell 10（進階）：全 noweight 模型校準 ==========
# 如果 final_d 4模型的校準效果好，再用全部 noweight 模型試試看

RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
noweight_paths = sorted([
    os.path.join(RUNS_DIR, d)
    for d in os.listdir(RUNS_DIR)
    if 'noweight' in d
    and os.path.isdir(os.path.join(RUNS_DIR, d))
    and os.path.exists(os.path.join(RUNS_DIR, d, 'test_probs.npy'))
    and os.path.exists(os.path.join(RUNS_DIR, d, 'val_probs.npy'))
])
print(f'Total noweight runs for full calibration: {len(noweight_paths)}')

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []

cmd_all = [
    'python', 'src/calibrate.py',
    '--bert-runs', *noweight_paths,
    '--no-overlap-constraint',
    '--tag', 'cal_all',
    '--method', 'both',
    *prior_flag,
]

print(f'\nRunning calibration on {len(noweight_paths)} runs...')
result = subprocess.run(cmd_all, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode == 0 and os.path.exists(CACHE_PATH):
    # LB estimate
    all_subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal_all*.csv')))
    if all_subs:
        subprocess.run(['python', 'src/estimate_lb.py', '--quiet'] + all_subs,
                       cwd=REPO_DIR)
