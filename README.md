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

## Project layout

```
.
├── Rule.md                              # competition rules
├── plan.md                              # original strategy + EDA findings
├── 0523Plan.md                          # day-2 detailed plan
├── README.md                            # this file
├── kaggle_trainset.csv                  # 12,994 labelled abstracts
├── kaggle_testset.csv                   # 1,444 abstracts (no labels)
├── kaggle_testset_submission.csv        # submission template
├── src/
│   ├── utils.py                         # label map, CV split, metric helpers
│   ├── eda.py                           # quick EDA
│   ├── deep_eda.py                      # k-NN, χ², co-occurrence analysis
│   ├── baseline_tfidf.py                # TF-IDF + LogReg 5-fold (CPU)
│   ├── train_bert.py                    # BERT fine-tune (single-label CE)
│   ├── train_bert_multilabel.py         # BERT fine-tune (multi-label BCE)
│   ├── train_bert_title_only.py         # BERT fine-tune on title only
│   ├── ensemble_predict.py              # combine runs → submission
│   └── calibrate_and_submit.py          # per-class logit calibration (deprecated, see below)
├── notebooks/
│   └── train_pubmedbert_colab.ipynb     # Colab driver notebook (A100)
└── outputs/
    ├── fold_assignment.csv              # frozen 5-fold split (seed=42)
    ├── oof_tfidf_logreg.npy             # baseline OOF probs
    ├── test_tfidf_logreg.npy            # baseline test probs
    ├── baseline_metrics.json
    └── submissions/                     # all generated submission CSVs
```

## Quickstart

### Local (CPU, baseline only)

```bash
python3 src/eda.py
python3 src/deep_eda.py
python3 src/baseline_tfidf.py
# -> outputs/submissions/submission_tfidf_baseline.csv  (OOF Macro F1 0.525)
```

### Colab A100 (main training)

```python
!git clone https://github.com/eric20041027/Data_Mining.git
%cd Data_Mining

# 訓練單一模型 5-fold
for fold in range(5):
    !python src/train_bert.py \
        --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
        --fold {fold} --seed 42 --epochs 4 --batch-size 32 --lr 2e-5 \
        --max-length 512 --class-weight none \
        --tag pubmedbert_noweight_seed42_fold{fold}

# Ensemble + submission
!python src/ensemble_predict.py \
    --bert-runs 'outputs/bert_runs/pubmedbert_noweight_seed*_fold*' \
    --no-overlap-constraint \
    --tag my_run
```

## LB progression (0521–0523)

| Submission | Strategy | OOF | LB |
|---|---|---|---|
| TF-IDF + LogReg + overlap constraint | baseline | 0.525 | 0.471 |
| TF-IDF + LogReg (unconstrained) | drop constraint | 0.525 | 0.523 |
| PubMedBERT balanced + constraint + calibration | v2 | 0.650 | 0.491 |
| PubMedBERT balanced raw (no constraint, no cal) | v6 | 0.640 | 0.635 |
| PubMedBERT noweight × 1 | drop class weight | 0.652 | 0.641 |
| PubMedBERT noweight × 2 seeds | add multi-seed | 0.653 | 0.643 |
| 3 noweight models | + BioBERT noweight | 0.657 | 0.644 |
| **4 noweight models + large** | **final_d** | **0.659** | **0.646** |
| 5 noweight (final_d + seed=7) | v14 | 0.660 | 0.644 |
| PubMedBERT BCE single | multi-label BCE | 0.696 | **0.596** ❌ |
| v17_no_large (SciBERT+DeBERTa, no large) | 5-model CE | 0.658 | 0.6455 |
| v16_7models (all CE) | 7-model CE | 0.659 | 0.6421 |

**Current best (LB confirmed): `final_d_4noweight` LB 0.64596** ★

### Phase 4 (0523) results — Multi-architecture + BCE

| Single model | OOF | LB (in ensemble) |
|---|---|---|
| SciBERT noweight | 0.6471 | adds ~0 LB |
| DeBERTa-v3 noweight | 0.6450 | adds ~0 LB |
| **PubMedBERT BCE multi-label** | **0.6957** | **0.596** (alone, OOF leak) |

**Key 0523 finding — BCE is an OOF trap.** Multi-label BCE training uses full-train multi-hot targets, which means a row's target includes labels from sibling rows in OTHER folds. OOF rocketed to 0.6957, but LB tanked to 0.596 (gap −0.100, the worst we've seen). The BCE+ensemble (v18) was abandoned without submission given the obvious leak signature. To use BCE legitimately we'd need GroupKFold-by-text-hash to keep sibling rows in the same fold.

**Architectural diversity ceiling.** SciBERT + DeBERTa-v3 noweight (LB 0.6455 in v17) and PubMedBERT-large (LB 0.6460 in final_d) are interchangeable contributors. Combining them (v16, LB 0.6421) does NOT help — bigger ensembles dilute the signal. The 4-model `final_d` configuration appears to be the ceiling of PubMedBERT-family + standard CV.

## Key lessons learned

1. **Overlap constraint is harmful.** Restricting predictions to "labels observed in train for the same text" hurts because the test ground truth often picks a label outside the train-observed set. Always pass `--no-overlap-constraint`.
2. **Class weighting hurts.** Using `class_weight=balanced` over-suppresses the majority class (general pathological), which is in fact a "secondary" label that co-occurs ≥55% with the other four classes. Use `--class-weight none`.
3. **Post-hoc calibration overfits OOF.** A 5-dim logit bias search improved OOF +0.010 but lost −0.009 on LB. The OOF→LB gap is honest; don't over-engineer it.
4. **Same-architecture multi-seed has diminishing (eventually negative) returns.** 1→2 seed gave +0.0018 LB; 2→3 gave −0.0021 LB. Each new seed reinforces the same class-prediction bias.
5. **Architectural diversity > seed diversity.** Cross-arch ensemble (PubMedBERT + BioBERT + PubMedBERT-large) gains +0.005–0.010 LB per new model class.

## Compliance notes (Rule.md)

- All training uses only `kaggle_trainset.csv` labels.
- No external label sources, no test-label probing, no manual label inspection.
- Test set analyses use only **text features** (length, hash overlap, similarity, vocabulary) — never test labels.
- `overlap-constraint` (in `ensemble_predict.py`) maps **train labels** onto matching test texts; this is legitimate use of train labels, though we've disabled it because it hurt LB.
