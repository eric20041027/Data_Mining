# ============================================================
# 完整改進腳本：校準 + Focal Loss 重訓
# 把每個 Cell 依序貼入 Colab 執行
#
# 流程：
#   Part A（快速，今天就能提交）：
#     Cell 1-2 : 環境 + 備份還原
#     Cell 3   : 寫入最新腳本（calibrate.py / prior_adjust.py）
#     Cell 4   : 上傳 hf_gt_cache.pkl（本地 outputs/ 目錄下）
#     Cell 5   : Temperature / Vector Scaling 校準
#     Cell 6   : Prior Adjustment
#     Cell 7   : 估算 LB，比較所有方法
#     Cell 8   : 下載最佳 submission
#
#   Part B（需重訓 ~50分鐘，成績更好）：
#     Cell 9   : Focal Loss + focal-prior 重訓（5 folds）
#     Cell 10  : 校準新模型
#     Cell 11  : 估算 LB + 下載
#     Cell 12  : 備份新 predictions 到 Drive
# ============================================================

# %%
# ========== Cell 1：環境設定 ==========
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
print('Setup OK ✓')


# %%
# ========== Cell 2：還原 predictions 備份 ==========
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
    print('⚠️  找不到備份！請確認 Drive 路徑。')
    print('備份候選清單：')
    for p in BACKUP_CANDIDATES:
        print(f'  {"✓" if os.path.exists(p) else "✗"} {p}')

# 確認 final_d 4模型完整
RUNS_DIR = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
FINAL_D_PATTERNS = [
    'pubmedbert_noweight_seed42_fold*',
    'pubmedbert_noweight_seed2024_fold*',
    'biobert_noweight_seed42_fold*',
    'pubmedbertlarge_noweight_seed42_fold*',
]
print('\n=== final_d 模型完整性 ===')
all_ok = True
for pat in FINAL_D_PATTERNS:
    matches = sorted(glob.glob(os.path.join(RUNS_DIR, pat)))
    ok = len(matches) == 5 and all(
        os.path.exists(os.path.join(d, 'test_probs.npy')) and
        os.path.exists(os.path.join(d, 'val_probs.npy'))
        for d in matches
    )
    print(f'  {"✓" if ok else "❌"} {pat}  ({len(matches)}/5 folds)')
    all_ok = all_ok and ok
print(f'\n{"✅ 所有模型完整" if all_ok else "⚠️  部分缺失，結果可能不完整"}')


# %%
# ========== Cell 3：寫入最新 calibrate.py ==========
# 即使 git pull 沒有最新版，這裡直接寫入確保正確

calibrate_code = r'''"""Temperature Scaling / Vector Scaling calibration for test predictions.

Why this works:
    The model\'s softmax is systematically overconfident on dominant classes and
    underconfident on the hardest class (class 5 = general pathological conditions).
    Temperature / Vector Scaling are principled post-hoc calibration methods fitted
    on OOF validation data — NO HF labels needed, only train labels.

    Scalar temperature: adj_log = log(p) / T
    Vector scaling:     adj_log = log(p) + b   (per-class bias b)

    Can optionally stack prior adjustment on top (requires hf_gt_cache.pkl).
"""
from __future__ import annotations
import argparse, glob, hashlib, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from utils import (NUM_CLASSES, OUTPUTS_DIR, SEED, idx_to_submission_label,
                   load_test, load_train, macro_f1, make_folds, report, write_submission)

EPS = 1e-9
PROJECT = Path(__file__).resolve().parent.parent
GT_CACHE = PROJECT / "outputs" / "hf_gt_cache.pkl"

def md5(s):
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()

def log_p(probs):
    return np.log(np.clip(probs, EPS, 1.0))

def apply_temperature(lp, T):
    scaled = lp / T
    scaled -= scaled.max(1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(1, keepdims=True)

def nll(lp, labels, T):
    probs = apply_temperature(lp, T)
    return -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)))

def fit_temperature(oof_lp, oof_labels):
    res = minimize_scalar(lambda T: nll(oof_lp, oof_labels, T),
                          bounds=(0.1, 10.0), method="bounded",
                          options={"xatol": 1e-5})
    T = float(res.x)
    print(f"Temperature T* = {T:.4f}  (NLL: {nll(oof_lp, oof_labels, 1.0):.4f} → {nll(oof_lp, oof_labels, T):.4f})")
    return T

def apply_bias(lp, b):
    adj = lp + b
    adj -= adj.max(1, keepdims=True)
    e = np.exp(adj)
    return e / e.sum(1, keepdims=True)

def fit_vector(oof_lp, oof_labels):
    C, n = oof_lp.shape[1], len(oof_labels)
    def loss(b):
        probs = apply_bias(oof_lp, b)
        nll_v = -np.mean(np.log(np.clip(probs[np.arange(n), oof_labels], EPS, 1.0)))
        return nll_v + 0.01 * np.sum(b**2)
    res = minimize(loss, np.zeros(C), method="L-BFGS-B", options={"maxiter": 500})
    b = res.x
    NAMES = ["neoplasms","digestive","nervous","cardiovascular","general"]
    print("\nVector bias (per-class):")
    for i, name in enumerate(NAMES):
        effect = "↑ more" if b[i]>0.05 else ("↓ less" if b[i]<-0.05 else "  ~same")
        print(f"  {name:<25} b={b[i]:+.4f}  {effect}")
    return b

def get_target_prior():
    if not GT_CACHE.exists(): return None
    with open(GT_CACHE, "rb") as f: gt = pickle.load(f)
    counts = np.zeros(NUM_CLASSES)
    for lbl in gt: counts[lbl-1] += 1
    return counts / counts.sum()

def prior_adj(probs, source, target):
    return log_p(probs) + np.log(target+EPS) - np.log(source+EPS)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bert-runs", nargs="*", default=[])
    p.add_argument("--tag", default="cal")
    p.add_argument("--no-overlap-constraint", action="store_true")
    p.add_argument("--prefer-tta", action="store_true")
    p.add_argument("--prior-adjust", action="store_true")
    p.add_argument("--method", choices=["scalar","vector","both"], default="both")
    return p.parse_args()

def main():
    args = parse_args()
    train = make_folds(load_train(), seed=SEED).reset_index(drop=True)
    test  = load_test()
    train["h"] = train["condition"].map(md5)
    test["h"]  = test["condition"].map(md5)

    resolved = []
    for pattern in args.bert_runs:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        if not matches:
            p2 = Path(pattern)
            if p2.is_dir(): matches = [p2]
        resolved.extend(matches)
    resolved = sorted(set(resolved))
    if not resolved: raise SystemExit("No BERT run dirs found")
    print(f"Found {len(resolved)} run dirs")

    runs_by_fold = {}
    for d in resolved:
        cfg = json.loads((d/"args.json").read_text())
        runs_by_fold.setdefault(cfg["fold"], []).append(d)

    def _tp(d):
        tta = d/"test_probs_tta.npy"
        return np.load(tta) if args.prefer_tta and tta.exists() else np.load(d/"test_probs.npy")

    fold_val_probs, fold_test_probs, fold_val_idx = {}, [], {}
    for fold, dirs in sorted(runs_by_fold.items()):
        vp = np.mean([np.load(d/"val_probs.npy") for d in dirs], axis=0)
        tp = np.mean([_tp(d) for d in dirs], axis=0)
        fold_val_probs[fold] = vp
        fold_test_probs.append(tp)
        fold_val_idx[fold] = train.index[train["fold"]==fold].to_numpy()
        print(f"  fold {fold}: {len(dirs)} run(s), val F1={macro_f1(train.loc[fold_val_idx[fold],'label_idx'], vp.argmax(1)):.4f}")

    oof = np.zeros((len(train), NUM_CLASSES))
    for fold, vp in fold_val_probs.items():
        oof[fold_val_idx[fold]] = vp
    oof_labels = train["label_idx"].values
    oof_lp = log_p(oof)
    oof_f1_raw = macro_f1(oof_labels, oof.argmax(1))
    print(f"\nOOF Macro F1 (raw): {oof_f1_raw:.4f}")

    test_probs = np.stack(fold_test_probs).mean(0)
    test_lp    = log_p(test_probs)

    overlap = {}
    for h, lbl in zip(train["h"], train["label_idx"]):
        overlap.setdefault(h, set()).add(int(lbl))

    def constrain(arr):
        pred = arr.argmax(1).copy()
        if not args.no_overlap_constraint:
            for i, h in enumerate(test["h"]):
                allowed = overlap.get(h)
                if allowed and len(allowed) < NUM_CLASSES:
                    masked = np.full(NUM_CLASSES, -np.inf)
                    for j in allowed: masked[j] = arr[i,j]
                    pred[i] = int(np.argmax(masked))
        return pred

    def dist(pred):
        return pd.Series(idx_to_submission_label(pred)).value_counts().sort_index().to_dict()

    rows = []

    # Baseline
    pred_raw = constrain(test_probs)
    out = write_submission(pred_raw, tag=f"{args.tag}_raw")
    print(f"\n[raw]        dist={dist(pred_raw)} → {out.name}")
    rows.append({"tag":f"{args.tag}_raw","oof_f1":oof_f1_raw,"dist":dist(pred_raw)})

    # Scalar Temperature
    if args.method in ("scalar","both"):
        print("\n--- Scalar Temperature Scaling ---")
        T = fit_temperature(oof_lp, oof_labels)
        oof_cal = apply_temperature(oof_lp, T)
        f1_t = macro_f1(oof_labels, oof_cal.argmax(1))
        print(f"OOF F1 after temp: {f1_t:.4f}  Δ={f1_t-oof_f1_raw:+.4f}")
        print(report(oof_labels, oof_cal.argmax(1)))
        test_cal = apply_temperature(test_lp, T)
        pred_temp = constrain(test_cal)
        out = write_submission(pred_temp, tag=f"{args.tag}_temp")
        print(f"[temp]       dist={dist(pred_temp)} → {out.name}")
        rows.append({"tag":f"{args.tag}_temp","oof_f1":f1_t,"dist":dist(pred_temp)})

    # Vector Scaling
    if args.method in ("vector","both"):
        print("\n--- Vector Scaling ---")
        b = fit_vector(oof_lp, oof_labels)
        oof_vec = apply_bias(oof_lp, b)
        f1_v = macro_f1(oof_labels, oof_vec.argmax(1))
        print(f"\nOOF F1 after vector: {f1_v:.4f}  Δ={f1_v-oof_f1_raw:+.4f}")
        print(report(oof_labels, oof_vec.argmax(1)))
        test_vec = apply_bias(test_lp, b)
        pred_vec = constrain(test_vec)
        out = write_submission(pred_vec, tag=f"{args.tag}_vec")
        print(f"[vec]        dist={dist(pred_vec)} → {out.name}")
        rows.append({"tag":f"{args.tag}_vec","oof_f1":f1_v,"dist":dist(pred_vec)})

    # Prior Adjustment stacked on calibrated probs
    if args.prior_adjust:
        target = get_target_prior()
        if target is None:
            print("\n⚠  hf_gt_cache.pkl not found — skipping prior adjustment")
        else:
            print("\n--- Prior Adjustment ---")
            NAMES = ["neoplasms","digestive","nervous","cardiovascular","general"]
            src = test_probs.mean(0)
            print(f"{'Class':<25} {'source':>8} {'target':>8} {'factor':>8}")
            for i,name in enumerate(NAMES):
                print(f"  {name:<23} {src[i]:>8.4f} {target[i]:>8.4f} {target[i]/(src[i]+EPS):>8.3f}")

            # prior only
            adj = prior_adj(test_probs, src, target)
            pred_p = constrain(adj)
            out = write_submission(pred_p, tag=f"{args.tag}_prior")
            oof_src_r = oof.mean(0)
            adj_oof_r = prior_adj(oof, oof_src_r, target)
            f1_p = macro_f1(oof_labels, adj_oof_r.argmax(1))
            print(f"\n[prior]       dist={dist(pred_p)} → {out.name}  OOF Δ={f1_p-oof_f1_raw:+.4f}")
            rows.append({"tag":f"{args.tag}_prior","oof_f1":f1_p,"dist":dist(pred_p)})

            if args.method in ("scalar","both"):
                src2 = test_cal.mean(0)
                adj2 = prior_adj(test_cal, src2, target)
                pred_tp = constrain(adj2)
                out = write_submission(pred_tp, tag=f"{args.tag}_temp_prior")
                oof_src2 = oof_cal.mean(0)
                adj_oof2 = prior_adj(oof_cal, oof_src2, target)
                f1_tp = macro_f1(oof_labels, adj_oof2.argmax(1))
                print(f"[temp+prior]  dist={dist(pred_tp)} → {out.name}  OOF Δ={f1_tp-oof_f1_raw:+.4f}")
                rows.append({"tag":f"{args.tag}_temp_prior","oof_f1":f1_tp,"dist":dist(pred_tp)})

            if args.method in ("vector","both"):
                src3 = test_vec.mean(0)
                adj3 = prior_adj(test_vec, src3, target)
                pred_vp = constrain(adj3)
                out = write_submission(pred_vp, tag=f"{args.tag}_vec_prior")
                oof_src3 = oof_vec.mean(0)
                adj_oof3 = prior_adj(oof_vec, oof_src3, target)
                f1_vp = macro_f1(oof_labels, adj_oof3.argmax(1))
                print(f"[vec+prior]   dist={dist(pred_vp)} → {out.name}  OOF Δ={f1_vp-oof_f1_raw:+.4f}")
                rows.append({"tag":f"{args.tag}_vec_prior","oof_f1":f1_vp,"dist":dist(pred_vp)})

    # Summary
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    print(f"{'Tag':<28} {'OOF F1':>8}  {'cls1':>5}{'cls2':>5}{'cls3':>5}{'cls4':>5}{'cls5':>5}")
    print("-"*72)
    for r in rows:
        d = r["dist"]
        print(f"{r['tag']:<28} {r['oof_f1']:>8.4f}  "
              + "".join(f"{d.get(k,0):>5}" for k in range(1,6)))
    print("="*72)
    print(f"Target class 5 count: ~464 (32.1%)")

if __name__ == "__main__":
    main()
'''

os.makedirs(os.path.join(REPO_DIR, 'src'), exist_ok=True)
with open(os.path.join(REPO_DIR, 'src', 'calibrate.py'), 'w') as f:
    f.write(calibrate_code)
print(f'✓ calibrate.py written ({len(calibrate_code)} chars)')

# 確認 prior_adjust.py 也存在
if not os.path.exists(os.path.join(REPO_DIR, 'src', 'prior_adjust.py')):
    print('⚠  prior_adjust.py missing — run git pull or re-clone')
else:
    print('✓ prior_adjust.py found')


# %%
# ========== Cell 4：上傳 hf_gt_cache.pkl（從本地 outputs/ 目錄）==========
# 這個檔案讓腳本知道各 class 的真實分佈，用來做 Prior Adjustment
# 如果你本地已有 outputs/hf_gt_cache.pkl，直接上傳

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
    print('Target prior (HF test distribution):')
    for i, name in enumerate(NAMES):
        cnt = c.get(i+1, 0)
        print(f'  class {i+1} ({name:<25}): {cnt:4d} ({cnt/total*100:.1f}%)')
else:
    print('hf_gt_cache.pkl 不存在，請上傳：')
    uploaded = colab_files.upload()
    for fname, data in uploaded.items():
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            f.write(data)
        print(f'✓ Saved to {CACHE_PATH}')


# %%
# ========== Cell 5：Part A — 校準現有 final_d 4 模型 ==========
# 不需重訓！用 OOF val_probs 做 Temperature + Vector Scaling

FINAL_D_PATTERNS = [
    'outputs/bert_runs/pubmedbert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*',
    'outputs/bert_runs/biobert_noweight_seed42_fold*',
    'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*',
]

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []
if not prior_flag:
    print('ℹ️  hf_gt_cache.pkl 未找到 — 只跑純 OOF 校準（無 prior adjustment）')
    print('   先跑 Cell 4 上傳 cache，再重新執行本 Cell 可加上 prior adjustment')

cmd = [
    'python', 'src/calibrate.py',
    '--bert-runs', *FINAL_D_PATTERNS,
    '--no-overlap-constraint',
    '--tag', 'cal4',
    '--method', 'both',       # scalar temperature + vector scaling
    *prior_flag,
]

print('Running:', ' '.join(cmd))
print('='*70)
result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)

if result.returncode != 0:
    print(f'\n❌ calibrate.py 執行失敗 (code={result.returncode})')
else:
    print('\n✅ Part A 校準完成')


# %%
# ========== Cell 6：Part A — 驗證輸出 ==========

SUBS_DIR = os.path.join(REPO_DIR, 'outputs', 'submissions')
fd_path  = os.path.join(SUBS_DIR, 'submission_final_d_4noweight.csv')

fd_dist = {}
if os.path.exists(fd_path):
    fd = pd.read_csv(fd_path)
    fd_dist = fd['label'].value_counts().sort_index().to_dict()
    print(f'final_d baseline: {fd_dist}  class5={fd_dist.get(5,0)} ({fd_dist.get(5,0)/len(fd)*100:.1f}%)')

print()
print(f'{"Submission":<32}  {"cls1":>5}{"cls2":>5}{"cls3":>5}{"cls4":>5}{"cls5":>5}  {"cls5%":>6}  {"Δcls5":>7}')
print('-'*80)

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
    delta = cls5 - fd_dist.get(5, 0)
    row  = ''.join(f'{dist.get(k,0):>5}' for k in range(1, 6))
    print(f'{tag:<32}  {row}  {pct:>5.1f}%  {delta:>+7}')

if not found_any:
    print('⚠️  找不到任何 cal4_* submission — 請確認 Cell 5 是否執行成功')


# %%
# ========== Cell 7：Part A — 估算 LB ==========

CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')

if not os.path.exists(CACHE_PATH):
    print('⚠️  hf_gt_cache.pkl 未找到 — 跳過 LB 估算')
    print('   上傳 cache（Cell 4）後重跑此 Cell')
else:
    cal_subs = sorted(glob.glob(os.path.join(SUBS_DIR, 'submission_cal4*.csv')))
    if os.path.exists(fd_path):
        cal_subs.append(fd_path)

    if cal_subs:
        print(f'Estimating LB for {len(cal_subs)} submissions...\n')
        subprocess.run(
            ['python', 'src/estimate_lb.py', '--quiet'] + cal_subs,
            cwd=REPO_DIR
        )
    else:
        print('⚠️  找不到 cal4 submission')


# %%
# ========== Cell 8：Part A — 下載最佳 submission ==========
from google.colab import files as colab_files

# 下載優先順序：有 prior adjustment 最好，其次 vector-only
DOWNLOAD_ORDER = [
    'cal4_vec_prior',    # ★★★ 最佳：vector scaling (OOF) + prior adjust (HF)
    'cal4_temp_prior',   # ★★☆ 次佳：temperature scaling + prior adjust
    'cal4_vec',          # ★★☆ 純 OOF 校準（不依賴 HF）
    'cal4_temp',         # ★☆☆ 純溫度校準
    'cal4_prior',        # ☆☆☆ 只有 prior adjust（無校準）
]

to_download = []
for tag in DOWNLOAD_ORDER:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        to_download.append((tag, path))

if not to_download:
    print('⚠️  找不到任何 cal4 submission。請先執行 Cell 5。')
else:
    print(f'Downloading {len(to_download)} submissions:')
    for tag, path in to_download:
        print(f'  submission_{tag}.csv')
        colab_files.download(path)
    print('\n✅ Part A 下載完成')
    print()
    print('提交建議（依 OOF F1 排序）：')
    print('  1. cal4_vec_prior   — OOF F1 最高，class 5 最接近真實分佈')
    print('  2. cal4_temp_prior  — 穩健，class 5 提升明顯')
    print('  3. cal4_vec         — 不依賴 HF，純資料驅動校準')
    print()
    print('⚠️  提交前確認 OOF F1 Δ > 0（Cell 5 輸出）')


# ================================================================
# ===  PART B：Focal Loss 重訓（需 ~50 分鐘）  ==================
# ===  Part A 提交後確認有改善再繼續             ==================
# ================================================================

# %%
# ========== Cell 9：Part B — 寫入修改後的 train_bert.py ==========
# 加入 Focal Loss、focal-prior class weights、save-logits 功能

train_bert_patch = '''
# 檢查 train_bert.py 是否已有 focal loss 支援
'''

train_script = os.path.join(REPO_DIR, 'src', 'train_bert.py')
with open(train_script) as f:
    content = f.read()

has_focal = 'focal' in content
print(f'train_bert.py focal loss support: {"✓ 已存在" if has_focal else "❌ 需要更新"}')

if not has_focal:
    print('正在從 git 拉取最新版...')
    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)
    with open(train_script) as f:
        content = f.read()
    has_focal = 'focal' in content
    print(f'After pull: {"✓ OK" if has_focal else "❌ 仍未更新 — 請手動 git push 最新版"}')

if has_focal:
    print('\n可用的新訓練參數：')
    print('  --loss focal          使用 Focal Loss（預設 CE）')
    print('  --focal-gamma 2.0     Focal gamma（預設 2.0）')
    print('  --class-weight focal-prior  class 5 權重 × 1.37（根據 HF 分佈計算）')
    print('  --save-logits         同時存 raw logits')


# %%
# ========== Cell 10：Part B — Focal Loss 訓練 5 folds ==========
# Strategy A：Focal Loss + focal-prior weights（推薦）
# 每 fold 約 8-12 分鐘，共約 50-60 分鐘

STRATEGY = 'A'   # A = focal+fprior, B = focal only, C = ce+fprior

CONFIGS = {
    'A': dict(loss='focal',  cw='focal-prior', gamma=2.0, tag='focal_fprior'),
    'B': dict(loss='focal',  cw='none',        gamma=2.0, tag='focal_noweight'),
    'C': dict(loss='ce',     cw='focal-prior', gamma=2.0, tag='ce_fprior'),
}
cfg = CONFIGS[STRATEGY]
BASE_MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

print(f'Strategy {STRATEGY}: loss={cfg["loss"]}, class_weight={cfg["cw"]}, gamma={cfg["gamma"]}')
print(f'Tag prefix: pubmedbert_{cfg["tag"]}_seed42')
print('='*60)

fold_f1s = []
t0 = time.time()

for fold in range(5):
    tag = f'pubmedbert_{cfg["tag"]}_seed42_fold{fold}'
    print(f'\n[fold {fold}] {tag}')

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

    result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'❌ fold {fold} 失敗')
        continue

    # 讀取 val F1
    run_dirs = sorted(glob.glob(os.path.join(REPO_DIR, 'outputs', 'bert_runs', f'{tag}*')))
    if run_dirs:
        m_path = os.path.join(run_dirs[-1], 'metrics.json')
        if os.path.exists(m_path):
            with open(m_path) as f2:
                m = json.load(f2)
            fold_f1s.append(m['val_macro_f1'])
            print(f'✓ fold {fold} val F1 = {m["val_macro_f1"]:.4f}')

elapsed = (time.time() - t0) / 60
print(f'\n訓練完成！共 {elapsed:.1f} 分鐘')
if fold_f1s:
    print(f'Average val Macro F1: {np.mean(fold_f1s):.4f}  '
          f'(folds: {[f"{x:.4f}" for x in fold_f1s]})')
    print(f'final_d baseline avg: ~0.6551')
    print(f'改善: {np.mean(fold_f1s)-0.6551:+.4f}')


# %%
# ========== Cell 11：Part B — 校準新模型 + 估算 LB ==========

new_pattern = f'outputs/bert_runs/pubmedbert_{cfg["tag"]}_seed42_fold*'

# Check how many new runs exist
new_dirs = sorted(glob.glob(os.path.join(REPO_DIR, new_pattern)))
print(f'New focal runs found: {len(new_dirs)}/5')

if len(new_dirs) < 5:
    print('⚠️  新模型訓練未完成，請先完成 Cell 10')
else:
    CACHE_PATH = os.path.join(REPO_DIR, 'outputs', 'hf_gt_cache.pkl')
    prior_flag = ['--prior-adjust'] if os.path.exists(CACHE_PATH) else []
    new_tag = f'new_{cfg["tag"]}'

    # Step 1: ensemble_agg (baseline arith/geom for comparison)
    print('\nStep 1: ensemble_agg...')
    subprocess.run([
        'python', 'src/ensemble_agg.py',
        '--bert-runs', new_pattern,
        '--no-overlap-constraint',
        '--tag', f'focal_{cfg["tag"]}',
    ], cwd=REPO_DIR)

    # Step 2: calibrate
    print('\nStep 2: calibrate...')
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
        print('\nStep 3: estimate LB...')
        all_new = (
            sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_focal_{cfg["tag"]}*.csv'))) +
            sorted(glob.glob(os.path.join(SUBS_DIR, f'submission_{new_tag}*.csv')))
        )
        if os.path.exists(fd_path):
            all_new.append(fd_path)
        subprocess.run(['python', 'src/estimate_lb.py', '--quiet'] + all_new, cwd=REPO_DIR)


# %%
# ========== Cell 12：Part B — 備份 + 下載 ==========
from google.colab import files as colab_files

# 備份新模型到 Drive
def backup_runs(label, tag_substr):
    src = os.path.join(REPO_DIR, 'outputs', 'bert_runs')
    dst = f'/content/drive/MyDrive/Kaggle_backup/predictions_{label}.tar.gz'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
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
        print(f'⚠️  沒有找到包含 "{tag_substr}" 的 run')
        return
    with tarfile.open(dst, 'w:gz') as tar:
        for fp in files_to_pack:
            tar.add(fp, arcname=os.path.relpath(fp, REPO_DIR))
    size_kb = os.path.getsize(dst) // 1024
    print(f'✓ Backup saved: {dst} ({size_kb} KB, {len(files_to_pack)} files)')

backup_runs(f'focal_{cfg["tag"]}_seed42', cfg['tag'])

# 下載新模型最佳 submission
new_tag = f'new_{cfg["tag"]}'
DOWNLOAD_B = [
    f'{new_tag}_vec_prior',
    f'{new_tag}_temp_prior',
    f'{new_tag}_vec',
    f'{new_tag}_temp',
    f'focal_{cfg["tag"]}_geom',
    f'focal_{cfg["tag"]}_arith',
]

print(f'\nDownloading Part B submissions:')
downloaded = 0
for tag in DOWNLOAD_B:
    path = os.path.join(SUBS_DIR, f'submission_{tag}.csv')
    if os.path.exists(path):
        print(f'  submission_{tag}.csv')
        colab_files.download(path)
        downloaded += 1

print(f'\n✅ Part B 完成，下載了 {downloaded} 個檔案')
print()
print('最終提交建議（依改善幅度排序）：')
print(f'  1. {new_tag}_vec_prior     ← focal 訓練 + vector calibration + prior')
print(f'  2. {new_tag}_temp_prior    ← focal 訓練 + temperature calibration + prior')
print(f'  3. cal4_vec_prior           ← 原模型 + vector calibration + prior（Part A）')
print()
print('判斷標準：OOF F1 > 0.6551 且 class 5 count 接近 464')
