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

### Phase 4 — Day 4 (2026-05-23): Multi-Architecture + BCE

#### EXP-017: SciBERT noweight
- **模型**：`allenai/scibert_scivocab_uncased`
- **設定**：5-fold, seed=42, epochs=4, batch=32, lr=2e-5, class-weight=none
- **OOF Macro F1**：**0.6471**（per-fold 0.664/0.653/0.633/0.653/0.633）
- **訓練時間**：~14 分/fold（總 ~70 分鐘）
- **解讀**：跟 BioBERT 同等級（0.6423），稍弱於 PubMedBERT noweight（0.6524）。CS+Bio 預訓練詞彙不同，主要價值在 ensemble 多樣性

#### EXP-018: DeBERTa-v3-base noweight
- **模型**：`microsoft/deberta-v3-base`
- **設定**：5-fold, seed=42, epochs=4, **batch=24, lr=1e-5**（DeBERTa lr 敏感）
- **OOF Macro F1**：**0.6450**（per-fold 0.653/0.641/0.637/0.657/0.636）
- **訓練時間**：~46 分/fold（DeBERTa-v3 disentangled attention 較慢，總 ~76 分鐘）
- **解讀**：跟其他 base 模型同水準，**架構完全不同**（disentangled attention vs BERT），ensemble 多樣性最佳來源

#### EXP-019: Multi-label BCE PubMedBERT ⭐ 重大突破
- **檔案**：[src/train_bert_multilabel.py](src/train_bert_multilabel.py)
- **方法**：把每個 train 文本的所有 observed labels 標為 multi-hot target（5-dim binary），loss 從 CE 換成 BCEWithLogitsLoss，`problem_type="multi_label_classification"`，inference 對 sigmoid 結果 normalize 後 argmax
- **設定**：5-fold, seed=42, epochs=4, batch=32, lr=2e-5
- **OOF Macro F1**：**0.6957** ⭐ — **單模型 +0.0433 vs PubMedBERT noweight (0.6524)**
- **Class 5 F1**：**0.5631**（vs CE noweight 0.4968, **+0.066**）
- **Class 5 recall**：**0.4682**（vs CE noweight 0.3964, **+0.072**）
- **訓練時間**：~166 秒/fold（總 ~14 分鐘，跟 noweight 一樣）
- **解讀**：
  - 是迄今**單一改動最大的 OOF 提升**（單模型超過所有 ensemble）
  - 解放了多標籤訊號：CE 強迫挑一個 label，BCE 學「所有 observed labels」
  - 跨 fold「同文本不同 label 共享 multi-hot target」可能帶來輕微 OOF 樂觀（不是 leak，但 OOF→LB gap 可能擴大）

#### EXP-020 (POSTPONED): Title-only model
- **檔案**：[src/train_bert_title_only.py](src/train_bert_title_only.py)
- **狀態**：腳本已 push，但 BCE 結果太強以致時間優先用在 ensemble 探索
- **預期 OOF**：0.55–0.58
- **改成**：時間允許再跑

#### EXP-021: 6 個 ensemble 候選的 OOF 對比
全部跑完後產生 6 個候選 submission，OOF 結果出乎意料：

| 候選 | 組成 | 模型數 | OOF | class 5 F1 | submission class 5 % |
|---|---|---|---|---|---|
| `final_bce_only` | BCE only (5-fold) | 1 | **0.6957** | **0.563** | **24.2%** |
| v15_6models | 4 noweight (final_d 基礎) + SciBERT | 5 | 0.6568 | 0.508 | 19.9% |
| v16_7models | v15 + DeBERTa | 6 | 0.6585 | 0.508 | 20.2% |
| v17_no_large | v16 砍 large | 5 | 0.6582 | 0.507 | 20.2% |
| v18_with_bce | v17 + BCE | 6 | 0.6624 | 0.513 | 20.1% |
| v19_all | v17 + large + BCE | 7 | 0.6610 | 0.513 | 20.3% |

**關鍵發現**：BCE 加進 5-model CE ensemble 後僅貢獻 +0.004 OOF（v17→v18），遠低於 BCE 單獨的 0.696。原因：
- 5 個 CE 模型分享同樣的「class 5 under-predict」bias
- BCE 是唯一不同 bias 的模型，平均後優勢被 5:1 稀釋
- **BCE alone 反而比 ensemble OOF 高 0.034**

提交決策完全翻轉：BCE-only 取代 ensemble 成為主打。

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
| 5/23 | 1 | `submission_final_bce_only` | 0.6957 | **0.59599** ❌ | BCE OOF 灌水 −0.100 gap |
| 5/23 | 2 | `submission_v17_no_large` | 0.6582 | 0.64553 | 跟 final_d 同 gap，新架構 OOF honest |
| 5/23 | 3 | `submission_v16_7models` | 0.6585 | 0.64210 | 加 large + 新架構反而 worse |
| 5/23 | 4 | `submission_final_bce_grouped_only` | 0.6902 | **0.59076** ❌ | GroupKFold 不是 leak 修法 |
| 5/23 | 5 | `submission_v22_scibert_replace_biobert` | 0.6573 | 0.64373 | BioBERT > SciBERT |

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

### 🆕 已驗證效果（0523）

6. **Multi-label BCE 訓練（不適合單標籤評分 task）**
   - 原始 BCE (StratifiedKFold): OOF 0.6957, LB **0.596**, gap −0.100
   - GroupKFold BCE（修了 multi-hot leak）: OOF 0.6902, LB **0.591**, gap **同樣 −0.099**
   - **GroupKFold 沒救 LB** — leak 不是主因
   - 真正原因：BCE sigmoid + normalize 後的「軟」分布在 argmax 時容易選錯，CE 的「硬」分布更適合單標籤評分
   - 教訓：multi-label dataset 被強制單標籤 task，CE + noweight 才是正解，BCE 是死路

7. **SciBERT noweight**（單模型 OOF 0.647，加 ensemble 後幾乎無 LB 增益）
8. **DeBERTa-v3 noweight**（單模型 OOF 0.645，同上幾乎無增益）

9. **新架構 ≤ PubMedBERT-large**
   - v17 (新架構替代 large): LB 0.6455
   - final_d (含 large): LB 0.6460
   - 差距 0.0004，新架構無法超越 large 的醫學域適配

10. **過大的 ensemble 反而退步**
    - v16 (7 models 全包): LB 0.6421
    - 比 final_d (4 models) 還低 0.0039
    - 模型間訊號相互稀釋

11. **BioBERT > SciBERT 在這個 task**
    - v22 (final_d 把 BioBERT 換 SciBERT): LB 0.6437
    - final_d (含 BioBERT): LB 0.6460
    - BioBERT 的 PubMed+PMC 預訓練比 SciBERT 的 CS+Bio 更貼合本資料

### 🤔 仍待驗證

1. ~~**GroupKFold BCE**~~ — 已驗證：LB 0.591，**沒救**（leak 不是 BCE 的問題）
2. **Title-only ensemble member**（時間允許再跑，預期收益微）
3. ~~**Pseudo-labeling**~~ — 0524 驗證：PL 單模型 OOF 0.6556（+0.003），加進 final_d ensemble (v23) OOF 0.6586 反而 −0.0006，**沒有實質貢獻**
4. **Smaller, focused ensemble**（只 PubMedBERT 系列，無 BioBERT）
5. **DeBERTa-v3-large noweight**（fold 0 = 0.6648，5-fold 進行中）
6. **TTA (Test-Time Augmentation)** 對 DeBERTa-large 模型（需 model checkpoint 在 disk，僅當 session 仍活著時可用）

### Phase 5 — Day 5 (2026-05-24): Pseudo-labeling + DeBERTa-large + TTA

#### EXP-022: Pseudo-labeling
- **腳本**：[src/make_pseudo_labels.py](src/make_pseudo_labels.py) + [src/train_bert_pseudo.py](src/train_bert_pseudo.py)
- **方法**：用 final_d 的 4 model ensemble 對 test 預測，每類取 top-80 最自信的 = 400 筆 pseudo-labels，加進 train 重訓 PubMedBERT noweight 5-fold
- **per-class top N 動機**：直接 threshold 0.80 會產生極度不平衡的 PL（neoplasms 192, cardio 110, general 77, digestive 5, **nervous 0**），會強化模型偏見
- **單模型 OOF**：0.6556（+0.0032 vs PubMedBERT noweight 0.6524）
- **v23 (final_d + PL) OOF**：0.6586（**比 final_d 0.6592 還低 0.0006**）
- **結論**：PL 對 ensemble 沒貢獻，weak class 的 PL signal 信心只有 0.55-0.65，帶噪音

#### EXP-023: DeBERTa-v3-large noweight (進行中)
- **模型**：`microsoft/deberta-v3-large`（350M）
- **設定**：batch=8, lr=1e-5, epochs=3, max_length=512, class-weight=none
- **fold 0**：val Macro F1 = 0.6648（19 分鐘）
- **預估 5-fold OOF**：0.650-0.660
- **狀態**：fold 1-4 訓練中

#### EXP-024: TTA (Test-Time Augmentation)
- **腳本**：[src/tta_predict.py](src/tta_predict.py) + ensemble_predict `--prefer-tta` 旗標
- **方法**：對既有訓練好的模型（需 checkpoint 在 disk）做 inference，每筆 test 在 5 種 augmentation 上預測：
  1. original（原始）
  2. drop_sentence（隨機丟 10% 句子）
  3. shuffle_body（打散非標題句子順序）
  4. truncate_head（保留前 85% 字元）
  5. truncate_tail（保留後 85% 字元）
- **限制**：只能對「本 Colab session 訓練的」模型做 TTA，因為輕量備份不含 model weights
- **預期增益**：+0.002-0.008 LB（per-model marginal contribution）

### 🏔️ 當前 LB 天花板

`final_d_4noweight` LB **0.64596** 似乎是 PubMedBERT-family + standard CV 的上限。要突破需要：
- 修 BCE 的 CV leak（GroupKFold）
- 或改用完全不同的訓練範式（pseudo-labeling, mixup, ...）

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
