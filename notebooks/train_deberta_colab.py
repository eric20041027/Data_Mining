# ============================================================
# Phase 1：DeBERTa-v3-large Fine-tune（最大模型多樣性增益）
#
# 預計時間：~120 分鐘（5 folds × ~24 min on A100）
# 預期 val F1：0.650–0.660（架構差異大，ensemble 增益最高）
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
import os, sys, subprocess, tarfile, json, time
import numpy as np

REPO_DIR = '/content/Data_Mining'

# Clone if missing, else pull latest
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone',
        'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'fetch', 'origin'], check=True)
subprocess.run(['git', '-C', REPO_DIR, 'reset', '--hard', 'origin/main'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

# DeBERTa-v3 tokenizer 需要 sentencepiece
subprocess.run(['pip', 'install', '-q', 'sentencepiece'], check=True)

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
# ========== Cell 3：訓練 DeBERTa-v3-large（5 folds）==========
MODEL = 'microsoft/deberta-v3-large'
TAG   = 'deberta_v3_large_seed42'

fold_f1s = []
t0_total = time.time()

for fold in range(5):
    print(f'\n{"="*60}')
    print(f'  DeBERTa-v3-large  Fold {fold}/4')
    print(f'{"="*60}', flush=True)
    t0 = time.time()

    cmd_str = (
        f'cd {REPO_DIR} && python src/train_bert.py'
        f' --model {MODEL}'
        f' --fold {fold}'
        f' --seed 42'
        f' --epochs 4'
        f' --batch-size 16'       # A100 可跑 16；T4 請改 8
        f' --lr 1e-5'             # 大模型用較小 LR
        f' --max-length 512'
        f' --weight-decay 0.01'
        f' --warmup-ratio 0.1'
        f' --loss ce'
        f' --class-weight balanced'
        f' --tag {TAG}_fold{fold}'
    )
    ret = os.system(cmd_str)
    elapsed = (time.time() - t0) / 60
    status = '✅' if ret == 0 else f'❌ (exit {ret})'
    print(f'\n{status} Fold {fold} done ({elapsed:.1f} min)', flush=True)
    if ret != 0:
        print('⚠ 訓練失敗，請複製上面錯誤訊息回報')
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
print(f'DeBERTa-v3-large 訓練完成（{total_min:.1f} min）')
if fold_f1s:
    avg = np.mean(fold_f1s)
    print(f'  Per-fold F1: {[f"{f:.4f}" for f in fold_f1s]}')
    print(f'  Avg val F1:  {avg:.4f}')
    print(f'  Δ vs PubMedBERT noweight (0.6524): {avg - 0.6524:+.4f}')


# %%
# ========== Cell 4：快速驗證（DeBERTa single model）==========
import glob

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
CACHE    = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE) else []

cmd_cal = [
    'python', 'src/calibrate.py',
    '--bert-runs', f'outputs/bert_runs/{TAG}_fold*',
    '--no-overlap-constraint',
    '--method', 'both',
    '--tag', 'deberta_single',
    *prior_flag,
]
print('Calibrating DeBERTa single model...')
r = subprocess.run(cmd_cal, cwd=REPO_DIR, text=True)
print(f'{"✅" if r.returncode==0 else "❌"} calibrate done')

if os.path.exists(CACHE):
    subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_deberta_single*.csv')))
    subs += [os.path.join(SUBS_DIR, 'submission_cal4_vec_prior.csv')]
    result = subprocess.run(
        ['python', 'src/estimate_lb.py'] + subs,
        cwd=REPO_DIR, capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print('STDERR:', result.stderr[:300])
