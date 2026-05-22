# CHANGELOG — Medical Abstracts 5-Class Classification

完整實驗紀錄，按時間排序。用於最終比賽報告，探討每個嘗試的設計、執行與效果。

---

## 📌 Executive Summary

**任務**：醫學摘要單標籤分類，5 類，Macro F1 評分。
**資料集**：train 12,994 筆、test 1,444 筆（無標籤）。源自 Schopf et al. (2022) *Medical Abstracts TC*，原為多標籤但本競賽以單標籤評分。
**規則**：禁止使用任何外部標籤、禁止從 test 真實答案反推特徵。
**進度**：
- LB 起點（TF-IDF + 直覺 post-processing）：**0.471**
- 當前最佳：**0.64596**（4-model noweight ensemble + PubMedBERT-large）
- 0523 目標：≥ 0.655（透過多架構 + multi-label BCE）

**核心發現**：
1. 資料集本質為多標籤被強制單標籤化（2,400/10,395 unique 文本有 ≥2 個共存標籤）
2. Test 與 train 文本重疊 41%，但這個重疊**不**幫助預測（test ground truth 不一定在 train observed labels 內）
3. 移除 `class_weight=balanced` 是最大單一改進（+0.006 LB）
4. 多架構 ensemble > 多 seed ensemble，後者過度後反而負收益

---

## 🗓️ Experiment Timeline

### Phase 0 — Day 1 (2026-05-20): Foundation

#### EXP-001: Initial EDA
- **目的**：理解資料形態
- **發現**：
  - 5 類不平衡（最少 digestive 10.3%, 最多 general pathological 33.4%）
  - 平均文本 ~1,228 字元 / ~180 詞 / PubMedBERT token 中位數 230（>512 僅 1.2%）
  - Train 內部重複文本 2,599 筆、test 內部重複 23 筆、train↔test 重疊 589 unique 文本（佔 test unique 41%）
  - 2,400 個 unique 文本有 2–4 個不同 label（**完全平分，無 modal**）→ 多標籤資料集

#### EXP-002: TF-IDF + LogReg Baseline
- **設定**：ngram=(1,2), min_df=3, sublinear_tf, max_features=200k, LogReg balanced C=4.0, 5-fold StratifiedKFold seed=42
- **OOF Macro F1**：0.5252
- **LB（有 overlap constraint）**：0.47094
- **LB（無 overlap constraint）**：0.52257
- **發現**：overlap constraint 害 −0.052 LB；TF-IDF OOF→LB gap 僅 −0.003，OOF 完全 honest
- **檔案**：[src/baseline_tfidf.py](src/baseline_tfidf.py)

---

### Phase 1 — Day 2 (2026-05-21): BERT Exploration

#### EXP-003: PubMedBERT base + balanced class weights
- **模型**：`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- **設定**：5-fold, seed=42, epochs=4, batch=32, lr=2e-5, max_length=512, bf16, **class_weight=balanced**
- **OOF Macro F1**：0.6402
- **Class 5 F1 (general pathological)**：**0.43**（recall 只有 0.30 — 大缺陷）
- **檔案**：[src/train_bert.py](src/train_bert.py), [notebooks/train_pubmedbert_colab.ipynb](notebooks/train_pubmedbert_colab.ipynb)

#### EXP-004: Post-hoc per-class logit calibration
- **方法**：對 OOF 做 coordinate descent grid search 找 5-dim logit bias 使 OOF Macro F1 最大
- **發現的 bias**：[-0.89, -1.04, -0.80, -0.48, 0.00] → 等同「降低 class 5 預測門檻」
- **OOF 提升**：0.6402 → 0.6503 (+0.0101)
- **檔案**：[src/calibrate_and_submit.py](src/calibrate_and_submit.py)

#### EXP-005: First submission — v2 (calibration + overlap constraint)
- **設定**：EXP-003 + EXP-004 校正 + overlap-set 限制
- **submission class 5 比例**：31.4%（vs train 33.4%）
- **LB**：**0.49069** ❌
- **結論**：與 OOF 0.6503 落差 0.16，**極不正常**

#### EXP-006: Diagnostic — TF-IDF unconstrained
- **目的**：確認 overlap constraint 害人，還是 BERT OOF 灌水
- **LB**：0.52257（vs constrained 0.47094, **+0.052**）
- **結論**：**overlap constraint 是元凶**

#### EXP-007: BERT v6 raw (no constraint, no calibration)
- **設定**：EXP-003 模型，移除 constraint 和 calibration
- **LB**：**0.63467**（vs v2 的 0.49069, **+0.144**）
- **OOF→LB gap**：−0.005（healthy）
- **結論**：constraint 和 calibration 全要拋棄

#### EXP-008: BERT v5 (calibration, no constraint)
- **目的**：分離校正本身是否有效
- **LB**：0.62527（比 v6 raw 低 −0.009）
- **結論**：**校正會 overfit OOF**，永遠不要用

---

### Phase 2 — Day 3 (2026-05-22): No-weight + Ensemble

#### EXP-009: PubMedBERT noweight（移除 class weight）
- **修正**：`--class-weight none`（不再使用 balanced）
- **OOF Macro F1**：0.6524（+0.012 vs balanced 的 0.6402）
- **Class 5 F1**：**0.50**（vs balanced 的 0.43，**+0.065**）
- **Class 5 recall**：**0.40**（vs balanced 的 0.30，**+0.094**）
- **檔案**：同 [src/train_bert.py](src/train_bert.py) with `--class-weight none`

#### EXP-010: Multi-seed PubMedBERT noweight
- **訓練**：seed=42, 2024, 7（各 5-fold）
- **2-seed (42+2024) OOF**：0.6526（+0.0002 vs 1-seed）
- **3-seed (42+2024+7) OOF**：0.6555（+0.0029）
- **多 seed 主要在「降噪」**：OOF 微升但 LB 提升 +0.0018（cleaner predictions on fold variability）

#### EXP-011: BioBERT noweight
- **模型**：`dmis-lab/biobert-base-cased-v1.2`
- **單模型 OOF**：0.6423
- **加進 PubMedBERT × 2 seed 後 → v9 OOF (3-model)**：0.6469
- **驗證**：跨架構比同架構多 seed 更有效（同樣 +1 model，後者僅 +0.0030 OOF）

#### EXP-012: PubMedBERT-large noweight
- **模型**：`microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`
- **單模型 OOF**：~0.665（per-fold 0.673, 0.668, 0.650, 0.663, 0.642）
- **加進 ensemble 後增量**：+0.0026 OOF
- **觀察**：大模型 OOF→LB gap 較大（−0.013 vs base 的 −0.011），輕微 overfit OOF

#### EXP-013: 4-model noweight ensemble — `final_d`
- **組成**：PubMedBERT × 2 seeds + BioBERT × 1 + PubMedBERT-large × 1（共 20 runs）
- **OOF Macro F1**：0.6592
- **LB**：**0.64596** ⭐ **當前最佳**
- **檔案**：[outputs/submissions/submission_final_d_4noweight.csv](outputs/submissions/submission_final_d_4noweight.csv)

#### EXP-014: 完整 LB 4-model 對照
| 提交 | 模型 | OOF | LB |
|---|---|---|---|
| `final_a` | PubMedBERT noweight × 1 (seed=42) | 0.6524 | 0.64133 |
| `final_b` | + seed=2024 | 0.6526 | 0.64312 |
| `final_c` | + BioBERT noweight | 0.6566 | 0.64448 |
| **`final_d`** | **+ PubMedBERT-large** | **0.6592** | **0.64596** |

**增量分析**：
- a → b（多 1 seed 同架構）：OOF +0.0002, LB +0.0018
- b → c（多 1 architecture）：OOF +0.0040, LB +0.0014
- c → d（加 large 容量）：OOF +0.0026, LB +0.0015

#### EXP-015: v14 — 加入 PubMedBERT noweight seed=7
- **設定**：final_d + seed=7（共 25 runs）
- **OOF Macro F1**：0.6600（+0.0008 vs final_d）
- **LB**：0.64391 ❌（**−0.0021** vs final_d）
- **結論**：同架構多 seed 達極限，繼續加反而負收益。預期同架構 bias 過度強化。

---

### Phase 3 — Day 3 (晚): Deep EDA

#### EXP-016: 深度文本分析
- **檔案**：[src/deep_eda.py](src/deep_eda.py)
- **關鍵發現**：
  1. **Test 二極化**：41% 近重複 train（cosine ≥ 0.95）vs 56% 完全新（max sim < 0.3）。中間值幾乎不存在。
  2. **Class 5 是「次要標籤」**：與其他四類共現 55–71%（cardio↔general path 71.2% 最高）
  3. **每類強判別詞彙**：
     - neoplasms: cancer, tumor, carcinoma, breast, metastases
     - digestive: cirrhosis, colitis, bowel, hepatitis, crohn
     - nervous: brain, cerebral, seizures, dementia
     - cardiovascular: coronary, myocardial, hypertension
     - general path: respiratory, infections, asthma（**沒有核心 keyword**）
  4. **Title-only TF-IDF 5-fold OOF**：0.4784（弱訊號但 error pattern 不同，可作 ensemble 多樣性）
  5. **多標籤分布**：
     - 7,995 unique 文本只有 1 個 label
     - 2,400 unique 文本有 2–4 個 label（多數 2 個）
     - 平均 1.43 labels/text

---

### Phase 4 — Day 4 (2026-05-23, 規劃中)

#### EXP-017 (PLANNED): SciBERT noweight
- **模型**：`allenai/scibert_scivocab_uncased`
- **預期 OOF**：0.63–0.65
- **預期 LB（加進 ensemble）**：+0.005

#### EXP-018 (PLANNED): DeBERTa-v3-base noweight
- **模型**：`microsoft/deberta-v3-base`
- **設定**：batch=24, lr=1e-5（DeBERTa lr 敏感）
- **預期 OOF**：0.63–0.67
- **預期 LB（加進 ensemble）**：+0.005–0.015

#### EXP-019 (PLANNED): Multi-label BCE PubMedBERT
- **檔案**：[src/train_bert_multilabel.py](src/train_bert_multilabel.py)
- **方法**：把每個 train 文本的所有 observed labels 標為 multi-hot target（5-dim binary），loss 從 CE 換成 BCEWithLogitsLoss，`problem_type="multi_label_classification"`
- **動機**：直接 model 多標籤結構，不再強迫挑一個
- **預期單模型 OOF**：0.65–0.67
- **預期 ensemble 收益**：+0.005–0.015

#### EXP-020 (PLANNED): Title-only model
- **檔案**：[src/train_bert_title_only.py](src/train_bert_title_only.py)
- **方法**：只用文章第一句作 input
- **預期 OOF**：0.55–0.58
- **預期 ensemble 收益**：+0.001–0.003

---

## 🎯 LB Submissions Full Log

| 日期 | 順序 | Submission | OOF | LB | 備註 |
|---|---|---|---|---|---|
| 5/21 | 1 | `submission_pubmedbert_v2_calibrated` | 0.6503 | 0.49069 | constraint + cal，極差 |
| 5/21 | 2 | `submission_tfidf_baseline` | 0.5252 | 0.47094 | TF-IDF + constraint |
| 5/21 | 3 | `submission_pubmedbert_v6_raw` | 0.6402 | 0.63467 | 移除所有後處理 ✨ |
| 5/21 | 4 | `submission_tfidf_baseline_unconstrained` | 0.5252 | 0.52257 | TF-IDF 純預測 |
| 5/21 | 5 | `submission_pubmedbert_v5_calibrated_no_overlap` | 0.6503 | 0.62527 | 校正單獨負面 |
| 5/22 | 1 | `submission_final_d_4noweight` | 0.6592 | **0.64596** ⭐ | 當前 best |
| 5/22 | 2 | `submission_final_c_3noweight` | 0.6566 | 0.64448 | 不含 large |
| 5/22 | 3 | `submission_final_a_noweight42` | 0.6524 | 0.64133 | 單模型 noweight |
| 5/22 | 4 | `submission_final_b_pubmed_2seeds` | 0.6526 | 0.64312 | 2 seed PubMedBERT |
| 5/22 | 5 | `submission_v14_5noweight` | 0.6600 | 0.64391 | +seed=7 反而退步 |

---

## 🧠 Key Lessons Learned

### ❌ 不要做這些事

1. **Overlap constraint**（強迫 test 預測落在 train observed labels）
   - 動機：似乎合理 — 若 test 文本與 train 重複，應在 train labels 內
   - 實測：LB −0.05～−0.14（同時對 TF-IDF 和 BERT 都有害）
   - 為什麼錯：test ground truth 常是 train observed 之外的 label（多標籤資料集任一 label 可能被選為單標籤 truth）

2. **Class weight = balanced**
   - 動機：處理類別不平衡
   - 實測：majority class（general pathological 33.4%）反而 over-suppress，recall 從理想 0.6+ 跌到 0.30
   - 為什麼錯：general pathological 是「次要標籤」與其他類高度共現，balanced 假設它是「獨立類別」做錯了

3. **Per-class logit calibration after training**
   - 動機：根據 OOF macro F1 調整 bias
   - 實測：OOF +0.010 但 LB −0.009
   - 為什麼錯：5-dim bias 搜尋空間太大，OOF 樣本量不夠，必然 overfit

4. **同架構繼續加 seed（≥3 個）**
   - 動機：每 seed +0.001 LB
   - 實測：第 3 seed −0.002 LB（v14）
   - 為什麼錯：相關 ρ ~0.9，平均後過度強化同向 bias

5. **TF-IDF 混進 BERT ensemble（10% 權重）**
   - 動機：架構多樣性
   - 實測：LB −0.009
   - 為什麼錯：強弱模型差距 0.10，稀釋 BERT 訊號

### ✅ 確定有效的事

1. **移除 class_weight**（+0.006 LB）— 最大單一收益
2. **移除 overlap constraint**（+0.052 LB on TF-IDF, +0.144 on BERT）
3. **多 seed 1→2 個**（+0.0018 LB）— 降噪
4. **跨架構 ensemble**（+0.001–0.002 LB per new arch）
5. **PubMedBERT-large**（+0.0015 LB）

### 🤔 待驗證

1. **Multi-label BCE 訓練**（預期 +0.005–0.015）
2. **SciBERT / DeBERTa 新架構**（預期 +0.005 each）
3. **Title-only ensemble member**（預期 +0.001–0.003）

---

## 📂 Codebase Inventory

### 訓練腳本

| 檔案 | 用途 |
|---|---|
| [src/train_bert.py](src/train_bert.py) | 標準 BERT fine-tune（CE loss），支援 `--class-weight {balanced,none}` |
| [src/train_bert_multilabel.py](src/train_bert_multilabel.py) | Multi-label BCE 變體 |
| [src/train_bert_title_only.py](src/train_bert_title_only.py) | 只用標題的 BERT 變體 |

### 分析與基準

| 檔案 | 用途 |
|---|---|
| [src/utils.py](src/utils.py) | 共用工具（label mapping, CV split, metric, seed） |
| [src/eda.py](src/eda.py) | 基礎 EDA（分布、長度、token 統計） |
| [src/deep_eda.py](src/deep_eda.py) | 深度分析（k-NN, χ², 多標籤共現） |
| [src/baseline_tfidf.py](src/baseline_tfidf.py) | TF-IDF + LogReg baseline |

### 推論與 ensemble

| 檔案 | 用途 |
|---|---|
| [src/ensemble_predict.py](src/ensemble_predict.py) | 多模型 ensemble，支援 `--no-overlap-constraint` |
| [src/calibrate_and_submit.py](src/calibrate_and_submit.py) | Per-class logit calibration（**已棄用**，會 overfit） |

### Notebook

| 檔案 | 用途 |
|---|---|
| [notebooks/train_pubmedbert_colab.ipynb](notebooks/train_pubmedbert_colab.ipynb) | Colab driver（過時 — 用 0523Plan.md 內的 cell） |

### 規劃

| 檔案 | 用途 |
|---|---|
| [plan.md](plan.md) | Day 1 初版策略 + EDA |
| [0523Plan.md](0523Plan.md) | Day 4 詳細執行計劃 |
| [CHANGELOG.md](CHANGELOG.md) | 本檔案 |
| [README.md](README.md) | 專案總覽 |

---

## 🔬 Methodology Notes（給最終報告用）

### Cross-Validation 設計
- **5-fold StratifiedKFold**，seed=42，所有實驗共用同一份 `fold_assignment.csv`
- 為什麼不用 GroupKFold（按 text hash 分組）：實測 OOF→LB gap 已穩定在 −0.005～−0.013，OOF 是 honest 的，不需要 GroupKFold；同時 GroupKFold 會浪費「同文本不同 label」的多標籤訊號

### OOF→LB Gap 觀察
| 模型類型 | OOF→LB gap |
|---|---|
| TF-IDF + LogReg | −0.003 |
| BERT balanced 單模型 | −0.005 |
| BERT noweight 單模型 | −0.011 |
| 3 model ensemble | −0.012 |
| 4 model ensemble | −0.013 |

Gap 隨 ensemble 規模微增，可能是 model averaging 在 OOF 有輕微樂觀。

### Rule Compliance
- 所有 EDA 只用 test **文本特徵**（長度、hash、TF-IDF 相似度），**從未碰 test labels**
- Train labels 可合法使用（包括映射到 overlap test 文本）— 但實測顯示沒幫助
- 沒有任何 probing attack（提交不同 label 觀察 LB 變化反推答案）

---

## 🚀 Future Improvements（若有更多時間 / 配額）

### 短期（明天可做）
- [ ] **SciBERT noweight 5-fold**（60 分鐘，預期 +0.005 LB）
- [ ] **DeBERTa-v3-base noweight 5-fold**（70 分鐘，預期 +0.010 LB）
- [ ] **Multi-label BCE PubMedBERT**（60 分鐘，預期 +0.010 LB）
- [ ] **Title-only PubMedBERT**（30 分鐘，預期 +0.002 LB）

### 中期（時間允許）
- [ ] **Stacking meta-learner**：用 5-fold OOF probs 作 features 訓練 LR meta-model（風險：overfit）
- [ ] **Pseudo-labeling**：用高信心 test prediction 補進 train（風險：多標籤資料容易放大噪音）
- [ ] **Hard-example mining**：找 OOF F1 最低的 train 樣本，重訓時 up-weight
- [ ] **Knowledge distillation**：用大模型 ensemble 作 teacher，訓練小模型

### 沒做的非主流方向
- ❌ Manual rule-based override（違反 Rule 2 風險）
- ❌ External medical NER tools / UMLS lookup（外部知識，不違規但工程成本高）
- ❌ Adversarial training, augmentation（資料夠多，效益低）

---

## 📚 References

1. Schopf, T., Braun, D., & Matthes, F. (2022). Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches.
2. Gu, Y. et al. (2020). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing (PubMedBERT).
3. Lee, J. et al. (2019). BioBERT: a pre-trained biomedical language representation model for biomedical text mining.
4. Beltagy, I. et al. (2019). SciBERT: A Pretrained Language Model for Scientific Text.
5. He, P. et al. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.
