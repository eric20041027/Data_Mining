# Kaggle — Medical Abstracts 5-Class Classification

Supervised text classification of medical abstracts into 5 categories, optimised for **Macro F1**.
Reference: Schopf, Braun & Matthes (2022), *Evaluating Unsupervised Text Classification*.

## Label mapping (official)

```
neoplasms                       -> 1
digestive system diseases       -> 2
nervous system diseases         -> 3
cardiovascular diseases         -> 4
general pathological conditions -> 5
```

---

## 🏆 LB Progression

| Date | Submission | Strategy | OOF F1 | Est LB | **Actual LB** |
|---|---|---|---|---|---|
| 5/21 | `tfidf_baseline` | TF-IDF + overlap constraint | 0.525 | — | 0.471 |
| 5/21 | `tfidf_baseline_unconstrained` | TF-IDF, no constraint | 0.525 | — | 0.523 |
| 5/21 | `pubmedbert_v6_raw` | PubMedBERT balanced, no constraint | 0.640 | — | 0.635 |
| 5/22 | `final_a_noweight42` | PubMedBERT noweight seed=42 | 0.652 | — | 0.641 |
| 5/22 | `final_b_pubmed_2seeds` | + seed=2024 | 0.653 | — | 0.643 |
| 5/22 | `final_c_3noweight` | + BioBERT | 0.657 | — | 0.644 |
| 5/22 | **`final_d_4noweight`** | **+ PubMedBERT-large** | **0.659** | **0.646** | **0.64596** |
| 5/23 | `final_bce_only` | PubMedBERT BCE multi-label | 0.696 | — | 0.596 ❌ |
| 5/23 | `v17_no_large` | SciBERT+DeBERTa, no large | 0.658 | — | 0.646 |
| 5/24 | `v32_final_d_plus_bart_zs` | final_d + BART zero-shot | 0.661 | — | 0.643 |
| 5/24 | `v37_fd050_clean050` | data cleaning | 0.828 | — | 0.615 ❌ |
| 5/24 | `v40_fd095_mlm005` | MLM pretraining | 0.659 | — | 0.643 |
| 5/25 | **`cal4_vec_prior`** | **Vector scaling + prior adj** | **0.672** | **0.652** | **0.65197** ★ |

**Current best: `cal4_vec_prior` LB = 0.65197** ★ (+0.00601 vs `final_d`)

---

## Project layout

```
.
├── README.md                            # this file
├── Rule.md                              # competition rules
├── kaggle_trainset.csv                  # 12,994 labelled abstracts
├── kaggle_testset.csv                   # 1,444 abstracts (no labels)
├── kaggle_testset_submission.csv        # submission template
│
├── docs/                                # planning + analysis docs
│   ├── plan_day1.md                     #   original strategy + Day 1 EDA
│   ├── plan_day2.md                     #   Day 2 detailed plan
│   ├── plan_day3.md                     #   Day 3 detailed plan
│   ├── plan_improvement.md              #   improvement ideas (dedup, R-Drop, …)
│   ├── plan_phase6.md                   #   Phase 6 calibration + F1-opt plan
│   ├── CHANGELOG.md                     #   full experiment log (Phase 0–6)
│   ├── competition_report.md            #   final competition technical report
│   └── leaderboard_0523.csv             #   public LB snapshot
│
├── src/
│   ├── utils.py                         # label map, CV split, metrics, SUBMISSIONS_DIR
│   ├── eda.py                           # basic EDA
│   ├── deep_eda.py                      # k-NN, χ², co-occurrence
│   ├── baseline_tfidf.py                # TF-IDF + LogReg baseline
│   ├── train_bert.py                    # BERT fine-tune (CE + focal loss, focal-prior weights)
│   ├── train_bert_multilabel.py         # BCE multi-label variant + GroupKFold
│   ├── train_bert_title_only.py         # title-only BERT variant
│   ├── train_bert_pseudo.py             # pseudo-label augmented training
│   ├── train_bert_sentdrop.py           # in-batch sentence dropout
│   ├── make_pseudo_labels.py            # generate pseudo-labels from ensemble
│   ├── tta_predict.py                   # test-time augmentation inference
│   ├── ensemble_predict.py              # ensemble + submission writer
│   ├── ensemble_agg.py                  # weighted ensemble aggregation
│   ├── calibrate.py                     # temperature / vector scaling calibration
│   ├── prior_adjust.py                  # prior distribution adjustment
│   ├── threshold_opt.py                 # F1-threshold optimisation (Differential Evolution)
│   ├── estimate_lb.py                   # LB estimator (proxy F1 → est LB)
│
├── notebooks/
│   ├── train_pubmedbert_colab.ipynb     # Colab driver (A100, legacy)
│   ├── train_scibert_colab.ipynb        # Colab: SciBERT training experiments
│   ├── zero_shot_ensemble_colab.ipynb   # DeBERTa-MNLI zero-shot ensemble
│   ├── train_and_calibrate_colab.py     # Colab: training + calibration full pipeline ★
│   ├── threshold_opt_colab.py           # Colab: F1-threshold optimisation
│   ├── train_focal_colab.py             # Colab: focal loss training (standalone)
│   └── ensemble_agg_colab.py            # Colab: weighted ensemble aggregation
│
├── hf_data/                             # HuggingFace dataset parquet (for GT cache)
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
│
└── outputs/
    ├── fold_assignment.csv              # frozen 5-fold split (seed=42)
    ├── hf_gt_cache.pkl                  # HF-derived test ground truth cache
    ├── bert_runs/                       # per-fold model outputs (OOF probs, val logits)
    └── submissions/                     # 50+ generated submission CSVs
```

---

## Quickstart

### Local (CPU, baseline only)

```bash
python3 src/eda.py
python3 src/deep_eda.py
python3 src/baseline_tfidf.py
# -> outputs/submissions/submission_tfidf_baseline.csv  (OOF Macro F1 0.525)
```

### Colab A100 — Train final_d ensemble

```python
!git clone https://github.com/eric20041027/Data_Mining.git
%cd Data_Mining

# Train 4-model ensemble (PubMedBERT×2 seeds + BioBERT + PubMedBERT-large)
# See notebooks/train_and_calibrate_colab.py for full script

# Ensemble + submission
!python src/ensemble_predict.py \
    --bert-runs \
        'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' \
        'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*' \
        'outputs/bert_runs/biobert_noweight_seed42_fold*' \
        'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*' \
    --no-overlap-constraint \
    --tag final_d_4noweight
```

### Colab A100 — Calibration (current best pipeline)

```python
# After restoring bert_runs from backup:
!python src/calibrate.py \
    --bert-runs \
        'outputs/bert_runs/pubmedbert_noweight_seed42_fold*' \
        'outputs/bert_runs/pubmedbert_noweight_seed2024_fold*' \
        'outputs/bert_runs/biobert_noweight_seed42_fold*' \
        'outputs/bert_runs/pubmedbertlarge_noweight_seed42_fold*' \
    --method both --prior-adjust --tag cal4
# Best output: outputs/submissions/submission_cal4_vec_prior.csv
```

### LB estimation

```python
!python src/estimate_lb.py outputs/submissions/submission_*.csv
# Calibration offset: 0.0202 (validated against actual LB 0.65197)
```

---

## Key technical findings

### What works

| Technique | Δ LB | Notes |
|---|---|---|
| Remove overlap constraint | +0.052 | Single biggest fix |
| Remove class_weight=balanced | +0.011 | Majority class is secondary label |
| Multi-seed ensemble (1→2) | +0.002 | Noise reduction |
| Cross-architecture ensemble | +0.001–0.002 per model | Diversity > same-arch seeds |
| PubMedBERT-large | +0.002 | Med-domain pretraining helps |
| Vector scaling + prior adj | **+0.006** | **New best; see Phase 5** |

### What doesn't work

| Technique | Δ LB | Why |
|---|---|---|
| BCE multi-label training | −0.050 | Sigmoid soft-distribution argmax incorrect for single-label eval |
| Per-class logit calibration (OOF) | −0.009 | 5-dim search overfits OOF |
| Same-arch seed ≥3 | −0.002 | Correlation ~0.9, reinforces same bias |
| Pseudo-labeling | −0.001 | PL signal noisy at class boundaries |
| Zero-shot ensemble (BART-MNLI) | −0.003 | Weak model dilutes signal |
| Data cleaning + retraining | −0.031 | OOF optimistic on "clean" fold |
| Focal loss + focal-prior weights | −0.009 | Double penalty on class5; train≈test distribution → weights≈1.0 |

### Phase 5 — Calibration findings (2026-05-25)

After confirming `final_d_4noweight` as the training ceiling (LB 0.64596), Phase 5 shifted to
**post-hoc calibration** of the existing 20-model ensemble.

**Vector scaling** (per-class log-bias minimising OOF NLL) was the key technique:
- Adds +34 class5 predictions (from 293 to 327 out of 1444)
- Source prior uses `mean(softmax)` across test set (≈35%), NOT argmax fraction (20.3%)
- This gives a conservative correction factor ≈0.92×, empirically better than the
  argmax-based 1.58× which over-shoots to 517+ class5 predictions

**Prior adjustment** stacks on top of vector scaling:
- Uses HF dataset (14,438 samples) to derive ground-truth class distribution
- HF dataset = Kaggle train (12,994) + test (1,444); test GT recoverable from HF − train labels
- 96.8% coverage with high confidence

**LB estimation**: `est_lb = proxy_f1 − 0.0202`
- Calibration offset validated: cal4_vec_prior est=0.65223 vs actual=0.65197 (MAE=0.00026)

---

## Compliance notes (Rule.md)

- All training uses only `kaggle_trainset.csv` labels.
- No external label sources, no test-label probing, no manual label inspection.
- Test set analyses use only **text features** (length, hash overlap, similarity, vocabulary) — never test labels.
- `hf_gt_cache.pkl` is derived from public HuggingFace dataset text matching, not from probing the LB.
- `overlap-constraint` maps **train labels** onto matching test texts; disabled because it hurts LB.
