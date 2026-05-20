# Kaggle — Medical Abstracts 5-Class Classification

Supervised text classification of medical abstracts into 5 categories, optimised for **Macro F1**.
Reference: Schopf, Braun & Matthes (2022), *Evaluating Unsupervised Text Classification*.

See [`plan.md`](plan.md) for the full implementation plan and EDA findings.

## Label mapping (official)

```
neoplasms                       -> 1
digestive system diseases       -> 2
nervous system diseases         -> 3
cardiovascular diseases         -> 4
general pathological conditions -> 5
```

## Layout

```
.
├── Rule.md                              # competition rules
├── plan.md                              # implementation plan + EDA findings
├── kaggle_trainset.csv                  # 12,994 labelled abstracts
├── kaggle_testset.csv                   # 1,444 abstracts (no labels)
├── kaggle_testset_submission.csv        # submission template
├── src/
│   ├── utils.py                         # label map, CV split, metric helpers
│   ├── eda.py                           # quick EDA script
│   ├── baseline_tfidf.py                # TF-IDF + LogReg 5-fold (CPU)
│   ├── train_bert.py                    # PubMedBERT fine-tune (one fold)
│   └── ensemble_predict.py              # combine runs, write final submission
├── notebooks/
│   └── train_pubmedbert_colab.ipynb     # Colab driver notebook (A100)
└── outputs/
    ├── fold_assignment.csv              # frozen 5-fold split
    ├── oof_tfidf_logreg.npy             # baseline OOF probs
    ├── test_tfidf_logreg.npy            # baseline test probs
    ├── baseline_metrics.json
    └── submission_tfidf_baseline.csv    # safety-net submission
```

## Quickstart

### Local (baseline only)

```bash
python3 src/eda.py
python3 src/baseline_tfidf.py
# -> outputs/submission_tfidf_baseline.csv  (OOF Macro F1 ~0.525)
```

### Colab A100 (main training)

1. Clone this repo into Colab:
   ```python
   !git clone https://github.com/eric20041027/Data_Mining.git
   %cd Data_Mining
   ```
2. Open `notebooks/train_pubmedbert_colab.ipynb` and run cells.
3. After 5-fold training:
   ```bash
   python3 src/ensemble_predict.py \
       --bert-runs "outputs/bert_runs/pubmedbert_base_seed42_fold*" \
       --tag pubmedbert_v1
   ```

## Key EDA finding

The dataset is a **multi-label corpus forced into single-label format**: 2,400 unique
training texts appear with 2–4 distinct labels (perfectly split, no modal). 41% of
test texts exactly match a training text. The pipeline keeps duplicates intact for
training and applies an overlap-constraint at inference (restricting prediction to
the label set observed for that text in train) — both fully compliant with Rule.md.
