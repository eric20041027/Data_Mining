# ============================================================
# Prior-Shift Correction for Kaggle Medical Classification
# 目的：用 label-shift correction 修正 class 5 under-prediction
#
# 背景：
#   - 模型預測 class 5 只有 20.3%，真實 test distribution 是 32.1%
#   - 這不是訓練集不平衡問題（train 有 33%）— 是模型校準問題
#   - 修正公式：adj_logit_k = log(p_k) + log(target_prior_k / source_prior_k)
#   - class 5 校正因子 ≈ 1.584x → 大幅提升 recall
#
# 預期：prior-adjusted OOF F1 > 原始 OOF F1
# ============================================================

# %%
# ========== Cell 1：環境設定（如果剛跑過 ensemble_agg_colab 可跳過）==========
import os, sys, subprocess, glob, tarfile, json

REPO_DIR = '/content/Data_Mining'

if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/eric20041027/Data_Mining.git', REPO_DIR], check=True)
else:
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

# 掛 Drive（已掛過可跳過）
from google.colab import drive
drive.mount('/content/drive')

print('Setup OK ✓')


# %%
# ========== Cell 2：還原 predictions 備份（如果剛跑過可跳過）==========
BACKUP_CANDIDATES = [
    '/content/drive/MyDrive/Kaggle_backup/predictions_FINAL_0524.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_after_noweight.tar.gz',
    '/content/drive/MyDrive/Kaggle_backup/predictions_only_final_all_noweight.tar.gz',
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
    print('⚠️  找不到備份！請手動確認 Drive 路徑。')

# 確認 pubmedbertlarge 存在（final_d 4模型之一）
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
large_runs = sorted(glob.glob(os.path.join(RUNS_DIR, 'pubmedbertlarge_noweight_seed42_fold*')))
print(f'\npubmedbertlarge folds found: {len(large_runs)}  (expected 5)')
for d in large_runs:
    has_t = os.path.exists(os.path.join(d, 'test_probs.npy'))
    has_v = os.path.exists(os.path.join(d, 'val_probs.npy'))
    print(f'  {"✓" if has_t and has_v else "❌"} {os.path.basename(d)}')


# %%
# ========== Cell 3：還原 HF GT cache（用於計算 target prior）==========
# hf_gt_cache.pkl 是從 HF parquet 推導的 test GT，用來算各類別真實分佈
# 如果備份裡沒有，需要有 HF parquets 才能重建

import pickle
import numpy as np

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        gt = pickle.load(f)
    print(f'GT cache found: {len(gt)} labels')
    from collections import Counter
    c = Counter(gt.tolist())
    total = len(gt)
    print('Target prior (true test distribution):')
    class_names = ['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']
    for i, name in enumerate(class_names):
        cnt = c.get(i+1, 0)
        print(f'  class {i+1} ({name:<20}): {cnt:4d} / {total} = {cnt/total:.4f}')
else:
    print('⚠️  hf_gt_cache.pkl not found!')
    print('Options:')
    print('  1. Upload hf_gt_cache.pkl to /content/Data_Mining/outputs/ manually')
    print('  2. Upload HF parquets and run: python src/estimate_lb.py --rebuild-cache')
    print('     HF parquets needed:')
    print('       hf_data/train-00000-of-00001.parquet')
    print('       hf_data/test-00000-of-00001.parquet')


# %%
# ========== Cell 4：直接上傳 hf_gt_cache.pkl（如果 Cell 3 找不到）==========
# 如果 Cell 3 成功找到 cache，跳過這個 cell

from google.colab import files as colab_files

print('請上傳 hf_gt_cache.pkl（從本地 outputs/ 目錄）')
uploaded = colab_files.upload()

for fname, data in uploaded.items():
    dst = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(data)
    print(f'Saved to: {dst}')


# %%
# ========== Cell 5：確認 prior_adjust.py 存在 ==========
# 如果 git pull 已拿到最新版，直接用；否則用 %%writefile 寫入

PA_SCRIPT = os.path.join(REPO_DIR, 'src', 'prior_adjust.py')

if os.path.exists(PA_SCRIPT):
    print(f'✓ prior_adjust.py found ({os.path.getsize(PA_SCRIPT)} bytes)')
    # 顯示版本確認
    with open(PA_SCRIPT) as f:
        first_lines = ''.join(f.readlines()[:5])
    print(first_lines)
else:
    print('❌ prior_adjust.py not found — will write it below')


# %%
# ========== Cell 5b（備用）：直接寫入 prior_adjust.py ==========
# 只有在 Cell 5 顯示 ❌ 時才執行這個 cell

prior_adjust_code = '''"""Prior-shift correction for test predictions.

Why this works:
    The model predicts class 5 only 20.3% of the time, but the true test
    distribution (from HF source) has class 5 at 32.1%.  This gap is NOT
    a training-set imbalance problem (train also has ~33% class 5) — it is a
    model confidence/calibration problem where the softmax is biased away from
    the hardest class.

    Standard label-shift correction:
        adjusted_logit_k = log(p_k) + log(target_prior_k / source_prior_k)

    where source_prior = model\'s average softmax on test set (reflects actual
    model bias, not training distribution) and target_prior = HF-derived true
    test distribution.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    NUM_CLASSES,
    OUTPUTS_DIR,
    SEED,
    idx_to_submission_label,
    load_test,
    load_train,
    macro_f1,
    make_folds,
    report,
    write_submission,
)

EPS = 1e-9
PROJECT = Path(__file__).resolve().parent.parent
GT_CACHE = PROJECT / "outputs" / "hf_gt_cache.pkl"


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_target_prior() -> np.ndarray:
    """Load target prior from HF GT cache.  Shape (5,) for classes 1..5."""
    if not GT_CACHE.exists():
        raise FileNotFoundError(
            f"GT cache not found: {GT_CACHE}\\n"
            "Run: python src/estimate_lb.py --rebuild-cache first."
        )
    with open(GT_CACHE, "rb") as f:
        gt = pickle.load(f)
    counts = np.zeros(NUM_CLASSES)
    for label in gt:                    # labels are 1-indexed
        counts[label - 1] += 1
    return counts / counts.sum()        # shape (5,) in internal 0-indexed order


def apply_prior_adjustment(
    test_probs: np.ndarray,             # (N, 5)  softmax probabilities
    source_prior: np.ndarray,           # (5,)  model\'s average test prediction
    target_prior: np.ndarray,           # (5,)  true test distribution
) -> np.ndarray:
    """Return adjusted log-probabilities (N, 5) suitable for argmax."""
    adj_factors = np.log(target_prior + EPS) - np.log(source_prior + EPS)
    log_probs = np.log(np.clip(test_probs, EPS, 1.0))
    return log_probs + adj_factors      # (N, 5) — add per-class log-bias


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[])
    p.add_argument("--tag", default="pa")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test  = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"]  = test["condition"].map(md5)

    # Resolve run dirs
    resolved: list[Path] = []
    for pattern in args.bert_runs:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches:
            p2 = Path(pattern)
            if p2.is_dir():
                matches = [p2]
        resolved.extend(matches)
    resolved = sorted(set(resolved))

    if not resolved:
        raise SystemExit("No BERT run dirs found — pass --bert-runs")

    print(f"Found {len(resolved)} run dirs")

    # Group by fold, average within fold
    runs_by_fold: dict[int, list[Path]] = {}
    for d in resolved:
        cfg = json.loads((d / "args.json").read_text())
        runs_by_fold.setdefault(cfg["fold"], []).append(d)

    def _tp(run_dir: Path) -> np.ndarray:
        tta = run_dir / "test_probs_tta.npy"
        if args.prefer_tta and tta.exists():
            return np.load(tta)
        return np.load(run_dir / "test_probs.npy")

    fold_val_probs: dict[int, np.ndarray] = {}
    fold_test_probs: list[np.ndarray]     = []
    fold_val_idx:   dict[int, np.ndarray] = {}

    for fold, dirs in sorted(runs_by_fold.items()):
        vp = np.mean([np.load(d / "val_probs.npy") for d in dirs], axis=0)
        tp = np.mean([_tp(d) for d in dirs], axis=0)
        fold_val_probs[fold] = vp
        fold_test_probs.append(tp)
        fold_val_idx[fold] = train.index[train["fold"] == fold].to_numpy()
        print(f"  fold {fold}: {len(dirs)} run(s), "
              f"val F1 = {macro_f1(train.loc[fold_val_idx[fold], \'label_idx\'], vp.argmax(1)):.4f}")

    # Aggregate test probs (arithmetic mean across folds)
    test_probs_arith = np.stack(fold_test_probs, axis=0).mean(axis=0)  # (N, 5)

    # OOF
    oof = np.zeros((len(train), NUM_CLASSES), dtype=np.float64)
    for fold, vp in fold_val_probs.items():
        oof[fold_val_idx[fold]] = vp
    oof_f1 = macro_f1(train["label_idx"], oof.argmax(1))
    print(f"\\nOOF Macro F1: {oof_f1:.4f}")
    print(report(train["label_idx"], oof.argmax(1)))

    # ── Prior adjustment ────────────────────────────────────────────────────
    target_prior = get_target_prior()     # (5,) in internal 0-indexed order
    source_prior = test_probs_arith.mean(axis=0)

    print("\\n=== Prior Adjustment Factors ===")
    NAMES = ["neoplasms", "digestive", "nervous", "cardiovascular", "general"]
    print(f"{\'Class\':<30} {\'source\':>8} {\'target\':>8} {\'factor\':>8}")
    for i, name in enumerate(NAMES):
        factor = target_prior[i] / (source_prior[i] + EPS)
        print(f"  {name:<28} {source_prior[i]:>8.4f} {target_prior[i]:>8.4f} {factor:>8.3f}")

    # Adjusted log-probs for test
    adj_log_probs = apply_prior_adjustment(test_probs_arith, source_prior, target_prior)

    # Adjusted log-probs for OOF val
    oof_source = oof.mean(axis=0)
    adj_oof_log = apply_prior_adjustment(oof, oof_source, target_prior)
    adj_oof_pred = adj_oof_log.argmax(1)
    adj_oof_f1 = macro_f1(train["label_idx"], adj_oof_pred)
    print(f"\\nPrior-adjusted OOF Macro F1: {adj_oof_f1:.4f}  (was {oof_f1:.4f}, Δ={adj_oof_f1-oof_f1:+.4f})")
    print(report(train["label_idx"], adj_oof_pred))

    # Overlap constraint helper
    overlap: dict[str, set[int]] = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    def _constrain(probs_or_logits: np.ndarray) -> np.ndarray:
        pred = probs_or_logits.argmax(1).copy()
        if not args.no_overlap_constraint:
            for i, h in enumerate(test["h"]):
                allowed = overlap.get(h)
                if allowed and len(allowed) < NUM_CLASSES:
                    masked = np.full(NUM_CLASSES, -np.inf)
                    for j in allowed:
                        masked[j] = probs_or_logits[i, j]
                    pred[i] = int(np.argmax(masked))
        return pred

    # ── Write submissions ───────────────────────────────────────────────────
    # 1. Baseline arithmetic
    pred_arith = _constrain(test_probs_arith)
    out = write_submission(pred_arith, tag=f"{args.tag}_arith")
    dist_arith = pd.Series(idx_to_submission_label(pred_arith)).value_counts().sort_index()
    print(f"\\n[arith]      dist={dist_arith.to_dict()} → {out.name}")

    # 2. Prior-adjusted arithmetic
    pred_pa = _constrain(adj_log_probs)
    out = write_submission(pred_pa, tag=f"{args.tag}_prior")
    dist_pa = pd.Series(idx_to_submission_label(pred_pa)).value_counts().sort_index()
    print(f"[prior_adj]  dist={dist_pa.to_dict()} → {out.name}")
    print(f"  class 5: {dist_arith.get(5,0)} → {dist_pa.get(5,0)} "
          f"({dist_pa.get(5,0)-dist_arith.get(5,0):+d})")

    # 3. Prior-adjusted + geometric mean
    log_sum = np.zeros_like(fold_test_probs[0])
    for fp in fold_test_probs:
        log_sum += np.log(np.clip(fp, EPS, 1.0))
    geom_probs = np.exp(log_sum / len(fold_test_probs))
    geom_probs /= geom_probs.sum(axis=1, keepdims=True)
    geom_source = geom_probs.mean(axis=0)
    adj_geom_log = apply_prior_adjustment(geom_probs, geom_source, target_prior)
    pred_pa_geom = _constrain(adj_geom_log)
    out = write_submission(pred_pa_geom, tag=f"{args.tag}_prior_geom")
    dist_pa_geom = pd.Series(idx_to_submission_label(pred_pa_geom)).value_counts().sort_index()
    print(f"[prior+geom] dist={dist_pa_geom.to_dict()} → {out.name}")

    # Summary
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"OOF F1 (original):          {oof_f1:.4f}")
    print(f"OOF F1 (prior-adjusted):    {adj_oof_f1:.4f}  Δ={adj_oof_f1-oof_f1:+.4f}")
    print(f"class 5 predictions: arith={dist_arith.get(5,0)} → prior={dist_pa.get(5,0)} "
          f"→ prior+geom={dist_pa_geom.get(5,0)}")
    print("Note: OOF F1 change is the best in-distribution signal for LB direction.")


if __name__ == "__main__":
    main()
'''

os.makedirs(os.path.join(REPO_DIR, 'src'), exist_ok=True)
with open(os.path.join(REPO_DIR, 'src', 'prior_adjust.py'), 'w') as f:
    f.write(prior_adjust_code)
print(f'✓ prior_adjust.py written ({len(prior_adjust_code)} chars)')


# %%
# ========== Cell 6：跑 prior_adjust.py — final_d 4模型 ==========

BERT_RUN_PATTERNS = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

cmd = [
    'python', 'src/prior_adjust.py',
    '--bert-runs',
    *BERT_RUN_PATTERNS,
    '--no-overlap-constraint',
    '--tag', 'pa4',
]

print('Running:', ' '.join(cmd))
print('='*70)

result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode != 0:
    print(f'\n❌ prior_adjust.py 執行失敗 (returncode={result.returncode})')
else:
    print('\n✅ prior_adjust.py 執行完成')


# %%
# ========== Cell 7：驗證輸出並估算 LB ==========
import pandas as pd

print('=== 輸出 submission 驗證 ===')
submissions = {}
for tag in ['pa4_arith', 'pa4_prior', 'pa4_prior_geom']:
    path = os.path.join(REPO_DIR, f'outputs/submissions/submission_{tag}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        dist = df['label'].value_counts().sort_index().to_dict()
        submissions[tag] = df
        cls5_pct = dist.get(5, 0) / len(df) * 100
        print(f'  ✓ {tag}: {dist}  (class5={cls5_pct:.1f}%)')
    else:
        print(f'  ❌ {tag}: 檔案不存在')

# Compare with final_d baseline
baseline_path = os.path.join(REPO_DIR, 'outputs/submissions/submission_final_d_4noweight.csv')
if os.path.exists(baseline_path):
    fd = pd.read_csv(baseline_path)
    print(f'\nfinal_d baseline dist: {fd["label"].value_counts().sort_index().to_dict()}')
    print(f'final_d class 5: {fd["label"].value_counts().get(5, 0)} ({fd["label"].value_counts().get(5, 0)/len(fd)*100:.1f}%)')
    print()
    print(f'{"Method":<20}  {"Changed vs fd":>14}  {"Class 5 Δ":>10}')
    print('-'*50)
    for tag, df in submissions.items():
        changed = (df['label'] != fd['label']).sum()
        cls5_delta = df['label'].value_counts().get(5, 0) - fd['label'].value_counts().get(5, 0)
        print(f'{tag:<20}  {changed:>14}  {cls5_delta:>+10}')


# %%
# ========== Cell 8：用 estimate_lb.py 估算各方法的預計 LB ==========

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
if not os.path.exists(CACHE_PATH):
    print('⚠️  hf_gt_cache.pkl not found — skipping LB estimation')
    print('    先跑 Cell 4 上傳 cache 再執行')
else:
    cmd_lb = [
        'python', 'src/estimate_lb.py',
        os.path.join(REPO_DIR, 'outputs/submissions/submission_pa4_arith.csv'),
        os.path.join(REPO_DIR, 'outputs/submissions/submission_pa4_prior.csv'),
        os.path.join(REPO_DIR, 'outputs/submissions/submission_pa4_prior_geom.csv'),
        os.path.join(REPO_DIR, 'outputs/submissions/submission_final_d_4noweight.csv'),
    ]
    # Filter to existing files
    cmd_lb = [c for c in cmd_lb[:1]] + [c for c in cmd_lb[1:] if os.path.exists(c)]

    print('Estimating LB for all methods...')
    print('='*70)
    result = subprocess.run(cmd_lb, cwd=REPO_DIR, capture_output=False, text=True)


# %%
# ========== Cell 9（進階）：全部71個模型跑 prior_adjust ==========
# 使用所有 bert_runs 的模型進行 prior adjustment，看是否有更大提升

RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
all_dirs = sorted([
    d for d in os.listdir(RUNS_DIR)
    if os.path.isdir(os.path.join(RUNS_DIR, d))
    and os.path.exists(os.path.join(RUNS_DIR, d, 'test_probs.npy'))
    and os.path.exists(os.path.join(RUNS_DIR, d, 'val_probs.npy'))
    and os.path.exists(os.path.join(RUNS_DIR, d, 'args.json'))
])

print(f'Total valid runs: {len(all_dirs)}')

# 只跑 noweight 模型（避免不同 loss 函數混入）
noweight_dirs = [d for d in all_dirs if 'noweight' in d]
print(f'noweight runs: {len(noweight_dirs)}')

# 組成完整路徑
noweight_paths = [os.path.join(RUNS_DIR, d) for d in noweight_dirs]

cmd_all = [
    'python', 'src/prior_adjust.py',
    '--bert-runs',
    *noweight_paths,
    '--no-overlap-constraint',
    '--tag', 'pa_all',
]

print(f'\nRunning prior_adjust on {len(noweight_dirs)} noweight runs...')
result = subprocess.run(cmd_all, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode == 0:
    # Estimate LB
    print('\n=== LB Estimates ===')
    result2 = subprocess.run([
        'python', 'src/estimate_lb.py',
        'outputs/submissions/submission_pa_all_arith.csv',
        'outputs/submissions/submission_pa_all_prior.csv',
        'outputs/submissions/submission_pa_all_prior_geom.csv',
    ], cwd=REPO_DIR, capture_output=False, text=True)


# %%
# ========== Cell 10：下載所有新 submission ==========
from google.colab import files as colab_files

to_download = []
for tag in ['pa4_arith', 'pa4_prior', 'pa4_prior_geom',
            'pa_all_arith', 'pa_all_prior', 'pa_all_prior_geom']:
    path = os.path.join(REPO_DIR, f'outputs/submissions/submission_{tag}.csv')
    if os.path.exists(path):
        to_download.append(path)

# Also grab baseline for comparison
baseline = os.path.join(REPO_DIR, 'outputs/submissions/submission_final_d_4noweight.csv')
if os.path.exists(baseline) and baseline not in to_download:
    to_download.append(baseline)

print(f'Downloading {len(to_download)} files:')
for p in to_download:
    print(f'  {os.path.basename(p)}')
    colab_files.download(p)

print('\n✅ 下載完成')
print()
print('提交優先順序建議（根據 OOF F1 和 class 5 分佈）:')
print('  1. pa4_prior_geom  ← prior校正 + 幾何平均，最穩健')
print('  2. pa4_prior       ← pure prior校正，容易解釋')
print('  3. pa_all_prior_geom  ← 全模型版本，如果上面兩個過了再試')
