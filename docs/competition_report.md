# Competition Report — Medical Abstracts 5-Class Classification

**課程競賽最終技術報告**

---

## 摘要

本競賽為醫學摘要 5 類單標籤分類任務，評分指標為 **Macro F1**。資料集來自 Schopf et al. (2022)，原本是多標籤資料集，被強制轉為單標籤評分。

- **起點 LB**：0.471（TF-IDF + overlap constraint）
- **最終 LB**：**0.65197**（Vector Scaling + Prior Adjustment）
- **總增益**：+0.181 LB
- **實驗週期**：5 天（2026-05-20 至 2026-05-25）
- **LB 提交次數**：~25 次（含 05-26 後續 5 次追加提交）
- **關鍵洞察**：資料集是多標籤強制單標籤化，移除所有「處理類別不平衡」的做法反而最有效

---

## 1. 任務與資料集分析

### 1.1 基本設定

| 項目 | 數值 |
|---|---|
| Train 筆數 | 12,994 |
| Test 筆數 | 1,444 |
| 類別數 | 5 |
| 評分指標 | Macro F1 |
| 平均文本長度 | ~1,228 字元 / ~180 詞 |
| PubMedBERT token 中位數 | 230（>512 僅 1.2%） |

### 1.2 類別分布

| 類別 | 名稱 | Train 比例 |
|---|---|---|
| 1 | neoplasms | 23.4% |
| 2 | digestive system diseases | 11.2% |
| 3 | nervous system diseases | 12.5% |
| 4 | cardiovascular diseases | 20.9% |
| 5 | general pathological conditions | 32.1% |

### 1.3 關鍵發現：多標籤資料集被強制單標籤化

這是本競賽最重要的洞察：

1. **Train 內部重複**：2,599 筆重複文本，其中 2,400 個 unique 文本擁有 2–4 個不同標籤（完全平分，無 modal）
2. **Train–Test 重疊**：589 個 unique 文本同時出現在 train 和 test（佔 test unique 的 41%）
3. **Class 5 是「次要標籤」**：General pathological conditions 與其他四類高度共現（cardio↔general 71.2%），並非獨立的主類

這些發現直接影響了「什麼有效、什麼無效」的決策：
- Class weight balanced 無效 → 大多數類的「不平衡」是多標籤共存，非真正的不平衡
- BCE multi-label 雖然 OOF 高，但 LB 差 → 多標籤訓練目標不適合單標籤評分
- Class 5 recall 天然低 → 模型在 class5 高度不確定（多標籤資料中 class5 幾乎「總是」共存的次要標籤）

---

## 2. 實驗歷程

### Phase 0 — Day 1（2026-05-20）：建立基礎

**EXP-001：EDA**
- 確認多標籤性質、文本重疊、類別分布
- 檔案：`src/eda.py`, `src/deep_eda.py`

**EXP-002：TF-IDF + LogReg**
- ngram=(1,2), min_df=3, LogReg balanced, 5-fold StratifiedKFold
- OOF F1：0.5252
- LB（有 constraint）：0.471 / LB（無 constraint）：0.523
- **發現**：overlap constraint 讓 LB 下降 0.052！

---

### Phase 1 — Day 2（2026-05-21）：BERT 初探

**EXP-003：PubMedBERT balanced**
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- 5-fold, epochs=4, batch=32, lr=2e-5, class_weight=balanced
- OOF：0.640，Class 5 recall：0.30（太低）

**EXP-004 & 005：Per-class logit calibration + overlap constraint**
- OOF 提升：0.640 → 0.650（+0.010）
- LB：**0.491** ❌（vs OOF 0.650，gap −0.159，**不正常**）
- 確認元凶：overlap constraint

**EXP-007：PubMedBERT v6 raw（移除所有後處理）**
- LB：**0.635**（vs v2 的 0.491，**+0.144**）
- **結論**：overlap constraint 和 calibration 全廢棄

**EXP-008：Calibration only（無 constraint）**
- LB：0.625（比 v6 raw 低 −0.010）
- **結論**：OOF calibration overfit，永遠不要用早期版本的 per-class logit search

---

### Phase 2 — Day 3（2026-05-22）：Class Weight 移除 + Ensemble

**EXP-009：PubMedBERT noweight**
- 移除 `class_weight=balanced` → `--class-weight none`
- OOF：0.6524（**+0.0122** vs balanced）
- Class 5 F1：0.50（+0.065），recall：0.40（+0.094）
- **這是整個競賽最大的單一改進**

**EXP-010 至 EXP-013：逐步 Ensemble 建構**

| 版本 | 組成 | OOF | Δ OOF | LB | Δ LB |
|---|---|---|---|---|---|
| final_a | PubMedBERT noweight seed=42 | 0.652 | baseline | 0.641 | baseline |
| final_b | + seed=2024 | 0.653 | +0.000 | 0.643 | +0.002 |
| final_c | + BioBERT noweight | 0.657 | +0.004 | 0.644 | +0.001 |
| **final_d** | **+ PubMedBERT-large** | **0.659** | **+0.002** | **0.646** | **+0.002** |
| v14 | + seed=7（多 1 個 PubMedBERT） | 0.660 | +0.001 | 0.644 | **−0.002** |

- **v14 的教訓**：同架構第 3 個 seed 開始負收益（相關性 ρ~0.9，過度強化同向 bias）
- `final_d_4noweight` 確立為 **4-model ensemble 基準**（LB 0.64596）

---

### Phase 3 — Day 3（晚）：深度 EDA

**深度 k-NN + χ² 分析**（`src/deep_eda.py`）：
- Test 二極化：41% 近重複 train（cosine ≥ 0.95）vs 56% 完全新（max sim < 0.3）
- Class 5 強判別詞：inflammatory, syndrome, disorder, autoimmune, manifestations
- 跨類混淆主要在 class5↔class4（cardiovascular diseases 與 general pathological 高共現）

---

### Phase 4 — Day 4（2026-05-23）：多架構 + BCE

**EXP-017：SciBERT noweight**（OOF 0.647，加入 ensemble 後 LB 幾乎無增益）

**EXP-018：DeBERTa-v3-base noweight**（OOF 0.645，同上）

**EXP-019：Multi-label BCE PubMedBERT ⭐ 假突破**
- 方法：每個文本的所有 observed labels 標為 multi-hot target，BCEWithLogitsLoss
- OOF：**0.6957**（+0.043 vs PubMedBERT noweight）
- LB：**0.596** ❌（gap −0.100，災難性）
- GroupKFold 修 leak 後 OOF 仍 0.690，LB 仍 0.591（gap −0.099）
- **結論**：BCE sigmoid 軟分布不適合單標籤 argmax 評分，不是 leak 問題

**Phase 4 OOF 對比（多模型 ensemble 嘗試）**：

| Ensemble | OOF | LB |
|---|---|---|
| v17_no_large（CE，新架構） | 0.6582 | 0.6455 |
| v16_7models（全包） | 0.6585 | 0.6421 |
| v22_scibert_replace_biobert | 0.6573 | 0.6437 |

**最終確認**：`final_d_4noweight` 是 PubMedBERT 家族 + CE 訓練的上限。

---

### Phase 5 — Day 5（2026-05-24）：Post-hoc 改進嘗試

**EXP-022：Pseudo-labeling**
- 用 final_d ensemble 預測 test，每類取 top-80 最自信的 400 筆
- PL 單模型 OOF：0.6556（+0.003）
- v23（final_d + PL）OOF：0.6586（**比 final_d 0.6592 還低**）
- **結論**：PL 對 ensemble 無貢獻，class boundary 處的 PL 信心只有 0.55–0.65，帶噪音

**EXP-024：DeBERTa-v3-large**
- fold 0 F1：0.6648（~20 分/fold）
- 完整 5-fold 後 est LB 未超越 final_d

**EXP-025：TTA（Test-Time Augmentation）**
- 方法：每筆 test 做 5 種 augmentation（original, drop_sentence, shuffle_body, truncate_head, truncate_tail）
- v25_deberta_tta est LB：0.650（未超越 cal4_vec_prior 的 0.652）

**Zero-shot ensemble（BART-MNLI）**
- v32_final_d_plus_bart_zs LB：0.643（−0.003 vs final_d）

---

### Phase 6 — Day 6（2026-05-25）：Calibration + F1 Threshold Optimisation + Strategy C

這是最終也是最成功的改進方向。

#### 6.1 新增腳本概覽

**`src/calibrate.py`**（核心校準腳本）
- Temperature scaling：找純量 T 使 OOF NLL 最小（T≈1.0 → 無效果）
- Vector scaling：找 5-dim log-bias `b` 使 OOF NLL 最小（L-BFGS-B + L2 regularization）
- Prior adjustment：`adj_log = log(p) + log(target_prior / source_prior)`
- 輸出：`{tag}_raw`, `{tag}_temp`, `{tag}_vec`, `{tag}_prior`, `{tag}_temp_prior`, `{tag}_vec_prior`

**`src/prior_adjust.py`**（先行版本，已被 calibrate.py 整合）
- Source prior：`mean(softmax(log_p), axis=0)`（使用**平均機率**，非 argmax 比例）
- Target prior：從訓練集標籤分布直接統計得出

**`src/threshold_opt.py`**（F1 門檻優化）
- 直接用 Differential Evolution + Nelder-Mead 最大化 OOF Macro F1
- 方法：`f1`（直接 F1 opt）, `vec+f1`（先 NLL vector scaling 再 F1 opt）
- Regularization：`0.05 * ||b||²` 防止 overfit OOF

**`src/estimate_lb.py`**（LB 估計器）
- `est_lb = proxy_f1 − 0.0202`
- Calibration offset 從 final_d（proxy=0.6664 → LB=0.64596）錨定
- 驗證：cal4_vec_prior est=0.65223 vs actual=0.65197（MAE=0.00026，極準確）

**`src/ensemble_agg.py`**（加權 ensemble 聚合）
- 支援 arithmetic / geometric mean 的加權組合

#### 6.2 LB 估計方法

以 OOF proxy F1 估算預期 LB，校準偏移量從已提交結果錨定：
- 錨定點：`final_d`（proxy=0.6664 → LB=0.64596，offset=0.0204）
- 公式：`est_lb = proxy_f1 − 0.0202`
- 驗證：cal4_vec_prior est=0.65223 vs actual=0.65197（誤差 0.00026）

此方法讓每次提交前可先估計預期 LB，大幅節省提交配額。

#### 6.3 Vector Scaling 分析

**Source prior 選擇（重要）**：
- 選項 A：`argmax fraction`（模型預測各類佔比）= 20.3% class5
- 選項 B：`mean probability`（softmax 均值）= 35.0% class5
- Target prior：32.1% class5

| Source prior | Class5 correction factor | Class5 predictions | Est LB |
|---|---|---|---|
| mean_prob（35% → 32%） | **0.92×**（保守校正） | **327** | **0.652** |
| argmax（20% → 32%） | 1.58×（激進校正） | 517 | 0.615 ❌ |

**結論**：mean probability source prior 在這個資料集上實驗性更優。原因：模型對 class5 具高不確定性（平均 P(class5)≈35% 即使 argmax 只有 20%），激進校正會嚴重過矯。

#### 6.4 Calibration 完整結果

| Submission | Class5 count | Proxy F1 | Est LB | Actual LB |
|---|---|---|---|---|
| `final_d_4noweight` | 293 | 0.6664 | 0.646 | 0.64596 |
| `cal4_raw` | 293 | 0.6664 | 0.646 | — |
| `cal4_temp` | 293 | 0.6664 | 0.646 | — |
| `cal4_vec` | 327 | 0.6706 | 0.650 | — |
| `cal4_prior` | 314 | 0.6695 | 0.649 | — |
| `cal4_vec_prior` | **314** | **0.6724** | **0.652** | **0.65197** ★ |

#### 6.5 F1-Threshold Optimisation 結果

直接最大化 OOF Macro F1（`threshold_opt.py`）：

| 方法 | Regularize | Est LB | vs cal4_vec_prior |
|---|---|---|---|
| `f1opt4_f1opt` | 0.05 | 退步 | −0.005 以上 |
| `vf1opt4_f1opt_prior` | 0.05 | 退步 | −0.005 以上 |
| `f1opt4r03_f1opt` | **0.3** | 0.64773 | **−0.00424** |
| `f1opt4r03_f1opt_prior` | **0.3** | 0.64858 | **−0.00339** |
| `vf1opt4r03_f1opt` | **0.3** | 0.64898 | **−0.00299** |
| `vf1opt4r03_f1opt_prior` | **0.3** | 0.65100 | **−0.00097** |
| `vf1opt4r03_vec_prior` | — | **0.65223** | ≈ 0（等於 cal4_vec_prior） |

**結論**：F1-threshold optimization 在所有 regularize 強度下均無法超越 vector scaling baseline。
`vf1opt4r03_vec_prior` 與 `cal4_vec_prior` 完全等價（確認 vec scaling 是天花板）。
OOF overfit 在這個問題上無法靠 regularize 解決，已放棄此方向。

#### 6.6 Strategy C：CE + Focal-Prior Class Weights

**策略 A**（focal loss + focal-prior）：已在 Phase 5 確認失敗（Δ = −0.0085）

**策略 C**：`--loss ce --class-weight focal-prior`（移除 focal loss，只保留 focal-prior weights）

- focal-prior weights = `TARGET_PRIOR / train_prior`（歸一化使平均值 ≈ 1）
- class5 weight：32.1% / 30.4% ≈ 1.06×（輕微上調）
- 結果：5-fold avg val F1 = **0.6522**（Δ = −0.0002 vs PubMedBERT noweight 0.6524）
- 校準後 est LB：全部低於 0.65197

**結論**：focal-prior weights 對 class5 的調整幅度不足（1.06×），等同無效。Train 分布本就接近 target prior，幾乎無校正空間。

---

## 3. 原始碼演進紀錄

### `src/utils.py`

| 版本 | 更新內容 |
|---|---|
| Phase 0 | label map, CV split, macro_f1 metric, set_seed |
| Phase 2 | write_submission 函數（初版，寫到 `outputs/`） |
| Phase 6 | **加入 `SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"`**，統一所有輸出路徑；修復 calibrate.py 輸出找不到的 bug |

### `src/train_bert.py`

| 版本 | 更新內容 |
|---|---|
| Phase 1 | 初版：PubMedBERT fine-tune, CE loss, balanced class weight |
| Phase 2 | 加入 `--class-weight none`, `--no-overlap-constraint` |
| Phase 4 | 多架構支援（BiomedNLP-large, DeBERTa, SciBERT） |
| Phase 6 | **加入 focal loss**（`--loss {ce,focal}`, `--focal-gamma 2.0`）<br>**加入 focal-prior class weights**（`--class-weight focal-prior`）<br>**加入 `--save-logits`**（儲存 `val_logits.npy`） |

### `src/ensemble_predict.py`

| 版本 | 更新內容 |
|---|---|
| Phase 2 | 初版：arithmetic mean of softmax probs, overlap constraint 可選 |
| Phase 3 | 加入 geometric mean、weighted ensemble |
| Phase 4 | 加入 `--prefer-tta` 旗標（搭配 `src/tta_predict.py`） |

### `src/calibrate.py`（Phase 6 新增）

完整溫度/向量縮放校準腳本。主要函數：

```python
def fit_temperature(oof_log_probs, oof_labels) -> float:
    # minimize_scalar NLL, bounds=(0.1, 10.0)

def fit_per_class_temperature(oof_log_probs, oof_labels) -> np.ndarray:
    # minimize NLL + 0.01*sum(b^2) via L-BFGS-B

def apply_prior_adjustment(probs, source_prior, target_prior) -> np.ndarray:
    adj_factors = np.log(target_prior + EPS) - np.log(source_prior + EPS)
    return np.log(np.clip(probs, EPS, 1.0)) + adj_factors
```

Source prior 使用 `mean(softmax, axis=0)`（平均機率）而非 argmax 比例。

### `src/threshold_opt.py`（Phase 6 新增）

直接最大化 OOF Macro F1：

```python
def fit_f1_thresholds(oof_lp, oof_labels, regularise=0.05) -> np.ndarray:
    def objective(b):
        pred = (oof_lp + b[np.newaxis, :]).argmax(1)
        f1 = macro_f1(oof_labels, pred)
        return -f1 + regularise * np.sum(b**2)

    bounds = [(-3.0, 3.0)] * C
    de_result = differential_evolution(objective, bounds, maxiter=800, seed=42)
    nm_result = minimize(objective, de_result.x, method="Nelder-Mead")
    return nm_result.x
```

方法：`f1`（直接 F1 opt）、`vec+f1`（vector NLL then F1 opt）

### `src/estimate_lb.py`（Phase 6 新增）

```python
CALIBRATION_OFFSET = 0.0202  # proxy_f1 - actual_lb

def evaluate(submission_path, gt, verbose=True):
    # 計算 proxy F1, est LB, per-class metrics, confusion matrix
    est_lb = proxy_f1 - CALIBRATION_OFFSET
```

### `src/prior_adjust.py`（Phase 6 新增）

- Target prior 從訓練集標籤分布直接統計得出
- 支援 `--method {add,multiply}` 應用調整
- 在 `calibrate.py` 加入 `--prior-adjust` 後已被整合

### `src/ensemble_agg.py`（Phase 6 新增）

- 加權幾何/算術 ensemble 聚合
- 支援 `--weights` 手動指定模型權重

---

## 4. Colab 腳本演進

| 腳本 | 功能 | 狀態 |
|---|---|---|
| `train_pubmedbert_colab.ipynb` | 早期訓練 driver | Legacy |
| `train_scibert_colab.ipynb` | SciBERT 訓練實驗 | Legacy |
| `zero_shot_ensemble_colab.ipynb` | DeBERTa-MNLI zero-shot ensemble | Legacy |
| `ensemble_agg_colab.py` | weighted ensemble 實驗 | 備用 |
| `train_and_calibrate_colab.py` | 訓練 + 校準完整流程 | 備用 |
| `threshold_opt_colab.py` | F1 threshold 優化（已確認無效） | 封存 |
| `train_focal_colab.py` | focal loss 獨立訓練 | 封存 |
| `train_focalprior_colab.py` | Strategy C：CE + focal-prior（已確認無效）| 封存 |
| **`train_deberta_colab.py`** | **DeBERTa-v3-large 訓練（Phase 1）** | **當前** |
| **`train_pseudo_colab.py`** | **Pseudo-labeling 訓練（Phase 2）** | **當前** |
| **`final_ensemble_colab.py`** | **全 stack ensemble 校準（Phase 3）** | **當前** |

---

## 5. 成績表

### 實際 LB（有確認的）

| Submission | 策略 | OOF F1 | Est LB | **Actual LB** |
|---|---|---|---|---|
| `tfidf_baseline` | TF-IDF + constraint | 0.525 | — | 0.471 |
| `tfidf_baseline_unconstrained` | TF-IDF, no constraint | 0.525 | — | 0.523 |
| `pubmedbert_v6_raw` | PubMedBERT balanced, no post-proc | 0.640 | — | 0.635 |
| `final_a_noweight42` | PubMedBERT noweight seed=42 | 0.652 | — | 0.641 |
| `final_b_pubmed_2seeds` | + seed=2024 | 0.653 | — | 0.643 |
| `final_c_3noweight` | + BioBERT | 0.657 | — | 0.644 |
| `final_d_4noweight` | + PubMedBERT-large | 0.659 | 0.646 | **0.64596** |
| `final_bce_only` | BCE multi-label | 0.696 | — | 0.596 ❌ |
| `v17_no_large` | SciBERT+DeBERTa ensemble | 0.658 | — | 0.646 |
| `v22_scibert_replace_biobert` | swap BioBERT→SciBERT | 0.657 | — | 0.644 |
| `v32_final_d_plus_bart_zs` | + BART zero-shot | 0.661 | — | 0.643 |
| `v37_fd050_clean050` | data cleaning | 0.828 | — | 0.615 ❌ |
| `v40_fd095_mlm005` | MLM pretraining | 0.659 | — | 0.643 |
| **`cal4_vec_prior`** | **Vec scaling + prior adj** | **0.672** | **0.652** | **0.65197** ★ |

### 預估 LB（est_lb = proxy_f1 − 0.0202）

| Submission | Proxy F1 | Est LB | 備註 |
|---|---|---|---|
| `cal4_raw` | 0.6664 | 0.6462 | 無校準（基準） |
| `cal4_temp` | 0.6664 | 0.6462 | T≈1.0，無效 |
| `cal4_vec` | 0.6706 | 0.6504 | Vector scaling only |
| `cal4_prior` | 0.6695 | 0.6493 | Prior adj only |
| `cal4_vec_prior` | **0.6724** | **0.6522** | **已確認 0.65197** |
| `f1opt4r03_f1opt` | 0.6679 | 0.6477 | F1-opt r=0.3（未提交，低於最佳） |
| `f1opt4r03_f1opt_prior` | 0.6688 | 0.6486 | F1-opt+prior r=0.3（未提交） |
| `vf1opt4r03_f1opt` | 0.6692 | 0.6490 | Vec+F1-opt r=0.3（未提交） |
| `vf1opt4r03_f1opt_prior` | 0.6712 | 0.6510 | **已提交 → 實際 LB 0.65095** ❌ |
| `cal4_temp_prior` | 0.6664 | 0.6462 | **已提交 → 實際 LB 0.64917** ❌ |
| `stc_fd_plus_vec` | — | — | **已提交 → 實際 LB 0.64808** ❌ |
| `stc_fd_plus_vec_prior` | — | — | **已提交 → 實際 LB 0.64674** ❌ |
| `stc_single_vec_prior` | — | — | **已提交 → 實際 LB 0.63356** ❌ |

---

## 6. 關鍵技術洞察

### 6.1 為什麼 Class 5 難預測？

Class 5（general pathological conditions）在資料集中扮演「次要標籤」角色，與其他四類的共現率達 55–71%。在多標籤被強制單標籤化的情境下：

- 模型平均 P(class5) ≈ 35%（高不確定性）
- 但 argmax 選 class5 只有 20.3%（遠低於 test 真實比例 32.1%）
- Class 5 recall：final_d 約 40%，cal4_vec_prior 約 44.6%

### 6.2 Post-hoc Calibration 為何這次有效？

Phase 1 的 per-class logit calibration 失敗（OOF overfit），Phase 6 的 vector scaling 成功，差異在：

| 方面 | Phase 1 calibration | Phase 6 vector scaling |
|---|---|---|
| 優化目標 | OOF Macro F1（直接目標） | OOF NLL（proxy） |
| 參數數量 | 5（logit bias，過多） | 5（同樣） |
| Regularization | 無 | L2 0.01 |
| 資料量 | 單模型 OOF（12,994） | 4-model ensemble OOF |
| 收益 | OOF +0.010 → LB −0.009 | OOF +0.006 → LB +0.006 |

**關鍵差異**：優化 NLL（proxy）比直接優化 F1 更不容易 overfit OOF 樣本。

### 6.3 LB 估計的可靠性

`est_lb = proxy_f1 − 0.0202` 的驗證：
- 錨定點：final_d proxy=0.6664 → LB=0.64596（offset=0.0204）
- 驗證點：cal4_vec_prior proxy=0.6724 → est=0.6522 vs actual=0.65197（error=0.00023）

OOF→LB 的 proxy gap（proxy F1 vs actual LB）：

| 模型 | Proxy F1 | LB | Gap |
|---|---|---|---|
| TF-IDF | 0.525 | 0.523 | −0.002 |
| BERT balanced（單模型） | 0.640 | 0.635 | −0.005 |
| BERT noweight（單模型） | 0.652 | 0.641 | −0.011 |
| 4-model ensemble | 0.666 | 0.646 | −0.020 |
| 4-model + vec_prior | 0.672 | 0.652 | −0.020 |

Gap 隨 ensemble 規模增大（約 −0.020 for 4-model ensemble），穩定後可作可靠估計。

---

## 7. 未來方向

### Phase 7 計畫：Full-Stack Ceiling（進行中）

**合法天花板手段全疊加**，以 3 個 Colab 腳本分 3 階段執行：

| 階段 | 腳本 | 手段 | 預估 LB |
|---|---|---|---|
| Phase 1 | `train_deberta_colab.py` | DeBERTa-v3-large（最大架構多樣性） | +0.005–0.010 |
| Phase 2 | `train_pseudo_colab.py` | Pseudo-labeling（confidence ≥ 0.80） | +0.003–0.007 |
| Phase 3 | `final_ensemble_colab.py` | 6 組模型全 stack 校準 | 疊加 |
| **目標** | | | **~0.660–0.665** |

### 已排除的方向（實驗驗證無效）

| 方向 | 結果 | 原因 |
|---|---|---|
| BCE multi-label | LB 0.596（gap −0.100） | sigmoid 軟分布不適合單標籤 argmax |
| 3+ 個同架構 seeds | −0.002 | 高相關性 ρ~0.9，訊號重複 |
| Zero-shot ensemble（BART-MNLI） | −0.003 | 弱模型稀釋強模型 |
| Data cleaning + retraining | 0.615（−0.03） | OOF leak |
| Per-class logit calibration（Phase 1）| −0.009 | OOF overfit |
| F1-threshold optimization | 全部退步 | OOF overfit，regularize 無解 |
| Strategy C：CE + focal-prior | Δ = −0.0002 | train≈target prior，無校正空間 |
| Focal loss + focal-prior | Δ = −0.0085 | 雙重懲罰 class5 |

---

## 8. 結論

本競賽最大的收穫是對「資料集性質」的正確理解：多標籤強制單標籤化導致 class 5 天然難預測，所有「處理不平衡」的技巧（balanced weight, BCE, focal loss）都反效果。

最終 pipeline：
1. **訓練**：PubMedBERT noweight × 2 seeds + BioBERT noweight + PubMedBERT-large（共 20 個 folds）
2. **後處理**：Vector scaling（NLL 最小化）+ Prior adjustment（訓練集標籤分布校準）
3. **LB 估計**：proxy_f1 − 0.0202

LB 從 0.471 提升至 0.65197，總增益 **+0.181**，其中：
- 移除 overlap constraint：+0.052
- 移除 class weight balanced：+0.011
- 多模型 ensemble：+0.011
- Post-hoc vector scaling + prior adjustment：+0.006

---

---

## Phase 7 補充（Full-Stack Ceiling）

### MLE Code Review 發現的主要風險

由 `ecc:mle-reviewer` 審查後，確認以下已知風險（競賽期間不影響結果，日後需修正）：

| 嚴重度 | 問題 | 狀態 |
|---|---|---|
| CRITICAL | `calibrate.py` / `threshold_opt.py` 重新呼叫 `make_folds` 而非讀 `val_index.csv`；若 sklearn/pandas 版本改變會靜默漂移 | 已知風險，目前 seed=42 固定可規避 |
| HIGH | OOF F1 是 in-sample 估計，無 holdout 驗證 regularize 強度 | 已知，threshold-opt 已放棄 |
| HIGH | Prior adjustment `source_prior` 不一致（OOF 用 oof mean，test 用 test mean）| 已知，影響輕微 |

---

### Phase 7 最終結果（2026-05-26）

後續追加提交 5 次，均未超越最佳分數：

| 提交 | 實際 LB | vs 最佳 |
|---|---|---|
| `vf1opt4r03_f1opt_prior` | 0.65095 | −0.00102 |
| `cal4_temp_prior` | 0.64917 | −0.00280 |
| `stc_fd_plus_vec` | 0.64808 | −0.00389 |
| `stc_fd_plus_vec_prior` | 0.64674 | −0.00523 |
| `stc_single_vec_prior` | 0.63356 | −0.01841 |

**結論**：`cal4_vec_prior`（0.65197）為本次競賽不可突破的實際天花板，所有後處理手段均無法進一步提升。DeBERTa-v3-large 因 GPU 配額耗盡未能修正 bf16 問題，偽標籤因 nervous system class 無樣本而放棄。

---

*Report last updated: 2026-05-26*
*Best LB: 0.65197（cal4_vec_prior）← 最終成績*
*Phase 7：GPU 配額耗盡，結案*
*Project: https://github.com/eric20041027/Data_Mining*
