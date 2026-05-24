# Medical Abstracts 5-Class Classification — Plan

## Context

Kaggle 競賽要求對醫學摘要做 5 類分類，採用 **Macro F1** 評分。官方資料集對應 Schopf et al. 2022 的 *Medical Abstracts TC*，但本競賽允許用 `kaggle_trainset.csv` 的標籤做監督式學習（Rule.md 已澄清）。目標是建立一個完整、可重現的訓練/推論 pipeline，並產出可提交的 `kaggle_testset_submission.csv`。

## Dataset Snapshot

- Train: **12,994** 筆（`label`, `condition`），無缺失
- Test: **1,444** 筆（`condition` only）
- 文本：英文醫學摘要，平均 ~1,228 字元（≈ 200–300 tokens，512 token 視窗多數可裝得下）
- 標籤分布（不平衡）：
  | label_str | label_id | count | ratio |
  |---|---|---|---|
  | general pathological conditions | 5 | 4,334 | 33.4% |
  | neoplasms | 1 | 2,837 | 21.8% |
  | cardiovascular diseases | 4 | 2,747 | 21.1% |
  | nervous system diseases | 3 | 1,743 | 13.4% |
  | digestive system diseases | 2 | 1,333 | 10.3% |

## Official Label Mapping (必須嚴格遵守)

```python
LABEL2ID = {
    "neoplasms": 1,
    "digestive system diseases": 2,
    "nervous system diseases": 3,
    "cardiovascular diseases": 4,
    "general pathological conditions": 5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
```

## Environment

- **執行環境**：Colab Pro + **A100 GPU**（40GB VRAM）
- **本地**：僅用於 EDA、baseline、最後 submission 檢查
- **資料流**：把 `kaggle_trainset.csv` / `kaggle_testset.csv` 上傳到 Google Drive（`/content/drive/MyDrive/Kaggle_new/`），Colab notebook 從那裡讀取
- **輸出**：訓練完的權重與 OOF 機率寫回 Drive，本地下載 submission 上傳 Kaggle

## Files

- 本地工作目錄：`/Users/smallfire/Desktop/Kaggle_new/`
- 輸入：`kaggle_trainset.csv`、`kaggle_testset.csv`、`kaggle_testset_submission.csv`
- 規則：`Rule.md`
- 預計新增：
  - `src/eda.ipynb` — 本地資料探索
  - `src/baseline_tfidf.py` — TF-IDF + LogReg baseline（本地 CPU 即可）
  - `notebooks/train_pubmedbert_colab.ipynb` — A100 主力訓練 notebook
  - `notebooks/predict_colab.ipynb` — 推論並產生 submission
  - `src/utils.py` — 共用：seed、CV split、label mapping、metric
  - `outputs/submission_<tag>.csv` — 各版本提交檔
  - `models/<run>/` — fold 權重（存在 Drive）

## Strategy Overview

採「baseline → 強模型 → ensemble」三階段，每階段都用同一套 5-fold StratifiedKFold（固定 seed），可比較且能直接平均成 ensemble。

### Phase 0 — EDA（已完成，重大發現）

- 文本：train 平均 1,228 字元、~180 詞；PubMedBERT token 中位數 230，>512 僅 1.2%
- 沒有 <50 字元的極短文本
- **重大：這是「被強制單標籤化」的多標籤資料集**
  - Train 12,994 列 = 10,395 個 unique 文本
  - 7,995 個文本只有 1 個標籤
  - 2,400 個文本有 2–4 個不同標籤，每個標籤出現恰好 1 次（完全平分）
  - **不要 dedup**：重複等同多標籤軟訊號，CE loss 自然會學到 ~50/50 機率
- **Test 與 train 文本重疊 589 筆（unique 中佔 41%）**
  - 推論時對 overlap 文本，把預測限制在「train 觀察過的標籤集合」內（合法：只用 train 標籤）
  - 對單標籤 overlap（一條 train 對應一個 label）→ 直接用該 label
  - 對多標籤 overlap → 在那幾個 label 之中取模型 argmax

### Phase 1 — Baseline：TF-IDF + Linear Model

目的：建立可信的下限（Macro F1 通常能到 0.75+），同時當作 ensemble 的多樣性來源。

- TF-IDF：`ngram_range=(1,2)`、`min_df=3`、`sublinear_tf=True`、`max_features=200k`
- 模型：`LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear")` 與 `LinearSVC` 各一份
- 5-fold StratifiedKFold（seed=42），輸出 OOF 預測機率（SVC 用 calibrated）
- 評估：Macro F1（per-class F1 也印出來檢視最弱類別）

### Phase 2 — 主力模型：PubMedBERT Fine-tune

**模型選擇**（A100 40GB 充裕，可同時用多個強模型）：
- 主力：`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`（PubMed abstracts 最佳 domain match）
- 第二：`microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`（大模型，A100 跑得動）
- 第三：`dmis-lab/biobert-base-cased-v1.2`（多樣性來源）
- 必要時 fallback：`emilyalsentzer/Bio_ClinicalBERT`

**訓練設定（A100 最佳化）**：
- `max_length=512`、truncation=True、dynamic padding（DataCollatorWithPadding）
- Optimizer: AdamW、lr=2e-5（base）/ 1e-5（large）、weight_decay=0.01
- Scheduler: linear warmup 10% + linear decay
- Batch size: **32**（base 模型）/ **16**（large 模型）— A100 40GB 跑得動
- Epochs: 4（早停 patience=2，monitor val Macro F1）
- Loss: **Cross-Entropy with class weights**（`weight = N / (K * count_k)`）以處理 digestive 10% 的不平衡
- Seed 列表：42、2024、7 — 跑 multi-seed 取平均
- 5-fold StratifiedKFold（與 Phase 1 同 split，方便 stacking）
- **bf16=True**（A100 原生支援，比 fp16 數值穩定）
- gradient checkpointing 視需要開（large 模型若 OOM 再開）
- 預估訓練時間：base × 5-fold × 3-seed ≈ 1.5–2 小時；large × 5-fold ≈ 1.5 小時

**Colab 注意事項**：
- 每完成 1 fold 就把權重與 OOF 機率寫回 Drive，避免 session 中斷重跑
- 用 `transformers.Trainer` + `save_strategy="epoch"` + `load_best_model_at_end=True`
- 訓練前先測試 1 fold 1 epoch 確認 pipeline 跑得通，再開全量

**驗證**：每 fold 存 best ckpt by Macro F1，輸出 OOF 機率。

### Phase 3 — Ensemble & Submission

- 主提交：5-fold PubMedBERT OOF 平均機率 → argmax
- 強化版：PubMedBERT × 3 seeds + BioBERT × 1 + TF-IDF LogReg，依 OOF Macro F1 加權平均
- 對「digestive system diseases」(class 2) 與「nervous system diseases」(class 3) 監測 per-class F1，若仍偏低則做 threshold tuning（softmax 後對該類乘以校正係數，用 OOF 搜尋最佳值）
- 產出 `kaggle_testset_submission.csv`：
  - `index` 維持 0..1443 不動
  - `label` 為 1–5 整數（依官方映射）
  - 提交前自動 sanity check：行數=1444、無 NaN、label ∈ {1,2,3,4,5}

## Critical Implementation Details

1. **Label 映射只在最後一步轉數字**：訓練時用 0–4（內部），輸出時 +0 或重映射成官方 1–5。集中在 `utils.py` 一處，避免散落。
2. **避免 data leakage**：
   - test 文本絕對不進 TF-IDF `fit`
   - 不做任何 test set 統計來調 hyperparameter（Rule.md 規則 2）
3. **隨機性控制**：固定 `random_state=42` 給 split，模型 seed 獨立記錄
4. **可重現**：每次 run 寫 `config.json` + git-style hash 到 `outputs/`
5. **CV 一致性**：所有模型用同一個 `fold_assignment.csv`（5 欄：sample_id, fold）

## Verification (End-to-End Test)

跑完計劃後驗收：

1. **EDA notebook 跑得通**，輸出長度/分布圖、找出 0 筆 train/test 重複（若有要回報）
2. **Baseline OOF Macro F1 ≥ 0.75**（合理下限；低於就先檢查 pipeline）
3. **PubMedBERT 單 fold val Macro F1 ≥ 0.82**（PubMed domain 模型在此資料集的合理水平）
4. **5-fold mean Macro F1 ≥ 0.83**，per-class F1 都 ≥ 0.70
5. **Submission 檢查**：
   ```bash
   python -c "
   import pandas as pd
   s = pd.read_csv('kaggle_testset_submission.csv')
   assert len(s) == 1444
   assert s['index'].tolist() == list(range(1444))
   assert s['label'].isin([1,2,3,4,5]).all()
   assert s['label'].isna().sum() == 0
   print(s['label'].value_counts().sort_index())
   "
   ```
6. **預測分布 sanity check**：test 預測的類別比例應大致接近 train 的比例（不應有某類完全不出現或壓倒性多數），否則回頭檢查 class weight 或 threshold。

## Open Questions / Risks

- Colab session 12 小時限制：每 fold 結束都把 ckpt + OOF 寫到 Drive，可中斷續跑
- 若 baseline 已 ≥ 0.85，PubMedBERT 邊際效益可能有限，evaluate 後再決定要不要全跑 3 seed
- HuggingFace 模型下載慢：首次下載完後快取存到 Drive (`os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"`)
- A100 配額（Colab Pro 有計算單位限制）：先用 small experiment 確認設定再開全量

## Execution Order

1. **本地**：建立 `src/utils.py`（label map、CV split、metric、seed）+ EDA + Baseline TF-IDF → 產生第一份保底 submission（~30 分鐘）
2. **上傳資料**：把 train/test/submission CSV、`utils.py`、`fold_assignment.csv` 上傳到 Drive
3. **Colab A100**：
   - 3a. 1 fold × 1 epoch 煙霧測試（確認 pipeline）
   - 3b. PubMedBERT-base 5-fold × seed=42（先看單 seed 結果）
   - 3c. 若 OOF Macro F1 達標再跑 seed 2024, 7
   - 3d. PubMedBERT-large 5-fold × seed=42（多樣性）
4. **本地或 Colab**：Ensemble + per-class 校正 → 最終 submission
5. 跑 verification 區塊所有 sanity check 後上傳 Kaggle
