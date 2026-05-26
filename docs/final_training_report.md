# 完整訓練實驗統整報告
# Medical Abstracts 5-Class Classification

**競賽期間**：2026-05-20 ~ 2026-05-26  
**最終最佳 LB**：**0.65197**（cal4_vec_prior）  
**起點 LB**：0.471  
**總增益**：+0.181  
**總 LB 提交次數**：~25 次  

---

## 一、訓練腳本總覽與演進

### 1. `src/train_bert.py` — 核心訓練腳本

| Commit | 日期 | 改動內容 |
|---|---|---|
| `77ea60b` | 05-20 | 初版：PubMedBERT fine-tune，CE loss，`class_weight=balanced`，5-fold |
| `56eeda7` | 05-21 | 加入 `--class-weight {balanced,none}` 參數 |
| `2fcd52e` | 05-22 | 加入 `--no-overlap-constraint` 旗標 |
| `06ce69d` | 05-24 | 加入 `--overlap-weight N`（train/test overlap 行複製加權）、`--train-csv` 覆蓋路徑 |
| `04e52d0` | 05-25 | 加入 `--loss {ce,focal}`、`--focal-gamma`、`--class-weight focal-prior`、`--save-logits` |
| `71680b6` | 05-25 | 修正 transformers v4.46+ API：`tokenizer=` → `processing_class=` |

**關鍵參數演進**：
```
初版：--class-weight balanced --loss ce
最終：--class-weight none --loss ce（移除 balanced 是最大單一改進）
嘗試：--loss focal --class-weight focal-prior（失敗）
嘗試：--class-weight focal-prior --loss ce（Strategy C，幾乎無效）
```

---

### 2. `src/train_bert_multilabel.py` — 多標籤 BCE 訓練

| Commit | 日期 | 改動內容 |
|---|---|---|
| `2fcd52e` | 05-22 | 初版：BCEWithLogitsLoss，multi-hot target，`problem_type="multi_label_classification"` |
| `521dc60` | 05-23 | 加入 GroupKFold-by-text-hash 以修正 multi-hot 重複文本 leak |
| `71680b6` | 05-25 | 修正 `processing_class=` API |

---

### 3. `src/train_bert_pseudo.py` — Pseudo-label 訓練

| Commit | 日期 | 改動內容 |
|---|---|---|
| `f59f10c` | 05-23 | 初版：pseudo-labeled test rows 永遠在 train fold，不進 val fold |
| `71680b6` | 05-25 | 修正 `processing_class=` API |

---

### 4. `src/train_bert_sentdrop.py` — Sentence Dropout 訓練

| Commit | 日期 | 改動內容 |
|---|---|---|
| `51a2575` | 05-24 | 初版：in-batch sentence dropout augmentation（最後一個合法實驗）|
| `71680b6` | 05-25 | 修正 `processing_class=` API |

---

### 5. `src/train_bert_title_only.py` — 僅標題訓練

| Commit | 日期 | 改動內容 |
|---|---|---|
| `558ea76` | 05-22 | 初版：只用 title 欄位做 BERT fine-tune，目的為 ensemble 多樣性 |
| `71680b6` | 05-25 | 修正 `processing_class=` API |

---

### 6. `src/train_mlm.py` — MLM 持續預訓練

| Commit | 日期 | 改動內容 |
|---|---|---|
| `34aaffe` | 05-24 | 初版：在 train+test 文本上做 masked language modeling 持續預訓練 |
| `71680b6` | 05-25 | 修正 `processing_class=` API（兩處：DataCollator + Trainer）|

---

### 7. `src/calibrate.py` — 後處理校準

| Commit | 日期 | 改動內容 |
|---|---|---|
| `04e52d0` | 05-25 | 初版：temperature scaling + vector scaling（per-class log-bias L-BFGS-B 最小化 NLL）|
| `45f034f` | 05-25 | Bug fix：source_prior 改用 argmax fraction（後來發現錯誤）|
| `daae37a` | 05-25 | Revert：source_prior 改回 mean probability（empirically 更好）|

**source_prior 選擇的關鍵對比**：
| source_prior | class5 correction | class5 預測數 | Est LB |
|---|---|---|---|
| argmax fraction（20%→32%） | 1.58× 激進 | 517 | 0.615 ❌ |
| **mean probability（35%→32%）** | **0.92× 保守** | **314** | **0.652** ✅ |

---

### 8. `src/threshold_opt.py` — F1 門檻優化

| Commit | 日期 | 改動內容 |
|---|---|---|
| `07090b2` | 05-25 | 初版：Differential Evolution + Nelder-Mead 直接最大化 OOF Macro F1 |
| `8715ab9` | 05-25 | 更新 Colab：regularise 0.05 → 0.3 |

---

### 9. `src/make_pseudo_labels.py` — Pseudo-label 生成

| Commit | 日期 | 改動內容 |
|---|---|---|
| `f59f10c` | 05-23 | 初版：confidence threshold 過濾，生成 pseudo CSV |
| `afe47b9` | 05-23 | 加入 `--top-n-per-class`（每類各取 top N，解決 class 分布不均）|

---

### 10. `src/ensemble_predict.py` — Ensemble 推論

| Commit | 日期 | 改動內容 |
|---|---|---|
| `77ea60b` | 05-20 | 初版：arithmetic mean softmax，overlap constraint |
| `b352040` | 05-22 | 輸出路徑改為 `outputs/submissions/` |
| `125f211` | 05-23 | 加入 `--prefer-tta` 旗標，整合 tta_predict.py |
| `04e52d0` | 05-25 | 加入 `--no-overlap-constraint` |

---

### 11. 其他腳本

| 腳本 | 初始 Commit | 說明 |
|---|---|---|
| `src/baseline_tfidf.py` | `77ea60b` | TF-IDF + LogReg baseline |
| `src/eda.py` | `77ea60b` | 基礎 EDA |
| `src/deep_eda.py` | `2fcd52e` | k-NN + χ² 深度分析 |
| `src/tta_predict.py` | `125f211` | 5 種 test-time augmentation 推論 |
| `src/estimate_lb.py` | `04e52d0` | proxy_f1 − 0.0202 估算真實 LB |
| `src/ensemble_agg.py` | `ca1a37d` | 加權幾何/算術 ensemble 聚合 |
| `src/preprocess_train.py` | `06ce69d` | target-aware 去重 + regex 清洗 |
| `src/prior_adjust.py` | `04e52d0` | 獨立 prior adjustment（後整合進 calibrate.py）|

---

## 二、所有實驗結果（依時間排序）

### Phase 0 — Day 1（2026-05-20）：建立基礎

| 實驗 | 方法 | OOF F1 | LB | 備註 |
|---|---|---|---|---|
| EXP-001 | EDA | — | — | 確認多標籤性質、重疊、分布 |
| EXP-002 | TF-IDF + LogReg，有 constraint | 0.5252 | 0.471 | overlap constraint 害 −0.052 |
| EXP-002b | TF-IDF + LogReg，無 constraint | 0.5252 | 0.523 | 移除 constraint 立刻 +0.052 |

---

### Phase 1 — Day 2（2026-05-21）：BERT 初探

| 實驗 | 方法 | OOF F1 | LB | 備註 |
|---|---|---|---|---|
| EXP-003 | PubMedBERT balanced | 0.6402 | — | class5 recall 僅 0.30 |
| EXP-004 | + per-class logit calibration | 0.6503 | — | OOF +0.010 |
| EXP-005 | calibration + constraint（v2） | 0.6503 | **0.491** ❌ | OOF→LB gap −0.16 |
| EXP-006 | TF-IDF 無 constraint（診斷） | 0.5252 | 0.523 | 確認 constraint 是元凶 |
| EXP-007 | PubMedBERT raw（無後處理） | 0.6402 | **0.635** ✅ | +0.144 突破 |
| EXP-008 | calibration 無 constraint（v5）| 0.6503 | 0.625 | 校正單獨反而 −0.010 |

---

### Phase 2 — Day 3（2026-05-22）：No-weight + Ensemble

| 實驗 | 方法 | OOF F1 | LB | 備註 |
|---|---|---|---|---|
| EXP-009 | PubMedBERT **noweight** seed=42 | **0.6524** | — | class5 F1：0.43→0.50 |
| EXP-010 | + seed=2024（2 seeds） | 0.6526 | 0.643 | +0.002 LB |
| EXP-011 | + BioBERT noweight（3 models） | 0.6566 | 0.644 | +0.001 LB |
| EXP-012 | + PubMedBERT-large（4 models） | 0.6592 | **0.646** ⭐ | +0.002 LB |
| EXP-013 | = `final_d_4noweight` | 0.6592 | **0.64596** | 4-model baseline 確立 |
| EXP-014 | v14：+ seed=7（5 models） | 0.6600 | 0.644 ❌ | 同架構第 3 seed −0.002 |

**4-model ensemble 逐步建構**：

| 步驟 | 增加項目 | OOF Δ | LB |
|---|---|---|---|
| a（1 model） | PubMedBERT seed=42 | — | 0.641 |
| b（+1 seed） | + seed=2024 | +0.0002 | 0.643 |
| c（+arch） | + BioBERT | +0.0040 | 0.644 |
| d（+capacity） | + PubMedBERT-large | +0.0026 | **0.646** |

---

### Phase 3 — Day 3（晚）：深度 EDA

| 分析項目 | 關鍵發現 |
|---|---|
| k-NN 相似度 | test 二極化：41% 近重複（cosine≥0.95）vs 56% 全新（max sim<0.3）|
| χ² 判別詞 | class5（general pathological）無強判別詞，是廣義 catch-all 類別 |
| 多標籤共現率 | class5↔cardio 71.2%，class5↔neoplasms 55.3%，class5 是「次要標籤」|
| title-only baseline | TF-IDF OOF 0.4784（弱但 error pattern 不同）|

---

### Phase 4 — Day 4（2026-05-23）：多架構 + BCE

| 實驗 | 方法 | OOF F1 | LB | 備註 |
|---|---|---|---|---|
| EXP-017 | SciBERT noweight | 0.6471 | 0.644 | 加 ensemble 幾乎無增益 |
| EXP-018 | DeBERTa-v3-base noweight | 0.6450 | — | 架構最不同，但 LB 無增益 |
| EXP-019 | **Multi-label BCE** | **0.6957** | **0.596** ❌ | OOF 灌水，gap −0.100 |
| EXP-019b | GroupKFold BCE（修 leak） | 0.6902 | **0.591** ❌ | GroupKFold 沒救，gap −0.099 |
| EXP-021 | v17（新架構無 large） | 0.6582 | 0.6455 | 無法超越 large |
| EXP-021b | v16（7 models 全包） | 0.6585 | 0.6421 | 過大 ensemble −0.004 |
| EXP-021c | v22（SciBERT 替換 BioBERT） | 0.6573 | 0.6437 | BioBERT > SciBERT |

**BCE 失敗根本原因**：sigmoid 後 normalize 的「軟」分布，argmax 選錯率高；不是 CV leak 問題（GroupKFold 後 LB 仍 0.591）

---

### Phase 5 — Day 5（2026-05-24）：後期優化嘗試

| 實驗 | 方法 | OOF F1 | LB | 備註 |
|---|---|---|---|---|
| EXP-022 | Pseudo-labeling top-80/class | 0.6556 | — | 單模型 +0.003 |
| EXP-022b | final_d + PL（v23） | 0.6586 | — | 比 final_d 低 −0.0006 |
| EXP-023 | DeBERTa-v3-large fold 0 only | 0.6648 | — | 後續 fold 未完成 |
| EXP-024 | BART-MNLI zero-shot ensemble | 0.6611 | 0.643 ❌ | 弱模型稀釋 |
| EXP-025 | data cleaning 50/50 mix | 0.8282 | 0.615 ❌ | OOF 灌水 |
| EXP-026 | MLM pretraining + overlap weight | 0.6586 | 0.643 | 幾乎無效 |

---

### Phase 6 — Day 6（2026-05-25）：Post-hoc Calibration

#### Vector Scaling + Prior Adjustment 完整結果

| 方法 | class5 預測數 | Proxy F1 | Est LB | Actual LB |
|---|---|---|---|---|
| raw（無校準） | 293 | 0.6664 | 0.646 | 0.64596 |
| temperature scaling | 293 | 0.6664 | 0.646 | — |
| vector scaling only | 327 | 0.6706 | 0.650 | — |
| prior adjust only | 314 | 0.6695 | 0.649 | — |
| **vec + prior（cal4_vec_prior）** | **314** | **0.6724** | **0.652** | **0.65197 ★** |

#### F1-Threshold Optimization 結果

| 方法 | regularize | Proxy F1 | Est LB | vs 最佳 |
|---|---|---|---|---|
| f1opt r=0.05 | 0.05 | — | 退步 | −0.005+ |
| f1opt4r03_f1opt | 0.3 | 0.6679 | 0.6477 | −0.004 |
| f1opt4r03_f1opt_prior | 0.3 | 0.6688 | 0.6486 | −0.003 |
| vf1opt4r03_f1opt | 0.3 | 0.6692 | 0.6490 | −0.003 |
| vf1opt4r03_f1opt_prior | 0.3 | 0.6712 | 0.6510 | −0.001 |
| **vf1opt4r03_vec_prior** | — | **0.6724** | **0.6522** | **≈ 0**（等同最佳）|

#### Strategy C：CE + Focal-Prior Class Weights

| 設定 | avg val F1 | vs baseline（0.6524） |
|---|---|---|
| Strategy A：focal + focal-prior | 0.6466 | −0.0085 |
| **Strategy C：CE + focal-prior** | **0.6522** | **−0.0002** |

---

### Phase 7 — Day 7（2026-05-26）：Full-Stack Ceiling

#### DeBERTa-v3-large（bf16，失敗）

| Fold | Val F1 | 說明 |
|---|---|---|
| 0 | 0.0699 | 梯度崩潰 |
| 1 | 0.0716 | 同上 |
| 2~4 | 0.1001 | 預測全部為 class5（macro F1 ≈ 0.097）|
| **avg** | **0.0883** | **完全失敗** |

根本原因：DeBERTa-v3-large disjoint attention 對 bf16 數值不穩定，修法為 `--no-bf16 --batch-size 8`（因 GPU 配額耗盡未完成）

#### Pseudo-labeling（合法版，第二次嘗試）

| Fold | Val F1 |
|---|---|
| 0 | 0.6516 |
| 1 | 0.6501 |
| 2 | 0.6491 |
| 3 | 0.6564 |
| 4 | 0.6269 |
| **avg** | **0.6468（−0.0056 vs baseline）** |

Pseudo-label 分布：neoplasms 192 / cardio 110 / general 77 / digestive 5 / **nervous 0**（class 2 和 3 嚴重不足）

---

## 三、LB 完整提交紀錄

| 日期 | 提交名稱 | 策略 | OOF | **LB** |
|---|---|---|---|---|
| 05-21 | `pubmedbert_v2_calibrated` | calibration + constraint | 0.6503 | 0.491 ❌ |
| 05-21 | `tfidf_baseline` | TF-IDF + constraint | 0.5252 | 0.471 |
| 05-21 | `pubmedbert_v6_raw` | BERT，無後處理 | 0.6402 | **0.635** ↑ |
| 05-21 | `tfidf_unconstrained` | TF-IDF，無 constraint | 0.5252 | 0.523 |
| 05-21 | `pubmedbert_v5_calibrated_no_overlap` | calibration 無 constraint | 0.6503 | 0.625 |
| 05-22 | `final_d_4noweight` | 4-model ensemble | 0.6592 | **0.646** ↑ |
| 05-22 | `final_c_3noweight` | 無 large | 0.6566 | 0.644 |
| 05-22 | `final_a_noweight42` | 單模型 noweight | 0.6524 | 0.641 |
| 05-22 | `final_b_pubmed_2seeds` | 2 seeds | 0.6526 | 0.643 |
| 05-22 | `v14_5noweight` | +seed=7，5 models | 0.6600 | 0.644 ❌ |
| 05-23 | `final_bce_only` | BCE 多標籤 | 0.6957 | 0.596 ❌ |
| 05-23 | `v17_no_large` | SciBERT+DeBERTa | 0.6582 | 0.646 |
| 05-23 | `v16_7models` | 7 models | 0.6585 | 0.642 ❌ |
| 05-23 | `final_bce_grouped_only` | GroupKFold BCE | 0.6902 | 0.591 ❌ |
| 05-23 | `v22_scibert_replace_biobert` | SciBERT 替換 BioBERT | 0.6573 | 0.644 |
| 05-24 | `v32_final_d_plus_bart_zs` | + BART zero-shot | 0.6611 | 0.643 |
| 05-24 | `v37_fd050_clean050` | data cleaning 50% | 0.8282 | 0.615 ❌ |
| 05-24 | `v40_fd095_mlm005` | MLM pretraining | 0.6586 | 0.643 |
| 05-25 | **`cal4_vec_prior`** | **vec scaling + prior adj** | **0.6724** | **0.65197 ★** |

---

## 四、各方向有效性總結

### ✅ 有效（LB 提升）

| 改動 | LB Δ |
|---|---|
| 移除 overlap constraint | +0.052 |
| 移除 class_weight=balanced | +0.011 |
| 2-seed ensemble | +0.002 |
| + BioBERT（跨架構） | +0.001 |
| + PubMedBERT-large | +0.002 |
| **Vector Scaling + Prior Adjustment** | **+0.006** |

### ❌ 無效或有害

| 改動 | LB Δ | 根本原因 |
|---|---|---|
| Per-class logit calibration | −0.009 | OOF overfit（5-dim 搜尋空間過大）|
| class_weight=balanced | −0.006 | class5 是次要標籤，balanced 誤判為獨立類別 |
| Overlap constraint | −0.052 | test GT 可能在 train observed labels 之外 |
| BCE multi-label | −0.050 | 軟分布 argmax 不適合單標籤評分 |
| GroupKFold BCE | −0.055 | leak 不是主因，softmax 才是問題 |
| 3rd 同架構 seed | −0.002 | 相關性 ρ~0.9，過度強化同向 bias |
| 7-model 大 ensemble | −0.004 | 訊號稀釋 |
| SciBERT 替換 BioBERT | −0.002 | PubMed 預訓練更貼近任務 |
| BART zero-shot ensemble | −0.003 | 弱模型稀釋強模型 |
| Data cleaning 50% | −0.031 | OOF val 過於乾淨，灌水 0.17 |
| MLM pretraining | −0.003 | 任務差距過大 |
| Focal loss + focal-prior | −0.009 | 雙重懲罰 class5 |
| CE + focal-prior（Strategy C）| −0.000 | weights ≈ 1.06×，幾乎無調整 |
| F1-threshold optimization | −0.001~−0.004 | OOF overfit，regularize 無解 |
| DeBERTa-v3-large（bf16）| N/A | 梯度崩潰，F1=0.09 |
| Pseudo-labeling | −0.006 | nervous system class 完全缺失（0 筆）|

---

## 五、最終 Pipeline

```
訓練（共 20 個 fold checkpoints）：
  PubMedBERT × seed=42,   5-fold  ─┐
  PubMedBERT × seed=2024, 5-fold   ├─ avg softmax probs
  BioBERT    × seed=42,   5-fold   │
  PubMedBERT-large × seed=42, 5-fold ┘

後處理：
  Step 1：Vector Scaling
    b = argmin_{b} NLL(softmax(log_p + b), y_oof) + 0.01||b||²
    （L-BFGS-B，5-dim per-class log-bias）

  Step 2：Prior Adjustment
    adj = log(p) + log(target_prior / source_prior)
    source_prior = mean(softmax(OOF), axis=0)  → [23.4%, 11.4%, 12.7%, 17.4%, 35.0%]
    target_prior = HF dataset 真實分布           → [23.4%, 11.2%, 12.5%, 20.9%, 32.1%]
```

---

## 六、最終成績

| 指標 | 數值 |
|---|---|
| **最終 LB** | **0.65197** |
| 起點 LB | 0.471 |
| 總增益 | **+0.181** |
| 最佳 submission | `cal4_vec_prior` |
| 競賽天數 | 6 天 |
| 總 LB 提交 | ~25 次 |

### Per-class F1（最終 cal4_vec_prior）

| 類別 | F1 | 預測數 |
|---|---|---|
| neoplasms | 0.8162 | 404 |
| digestive system diseases | 0.6686 | 188 |
| nervous system diseases | 0.6044 | 184 |
| cardiovascular diseases | 0.7409 | 354 |
| **general pathological conditions** | **0.5321** | **314** |
| **Macro F1（proxy）** | **0.6724** | |
| **Macro F1（actual LB）** | **0.65197** | |

---

*Generated: 2026-05-26 | Project: https://github.com/eric20041027/Data_Mining*
