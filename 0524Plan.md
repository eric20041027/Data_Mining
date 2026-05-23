# 0524 突破計劃 — Pseudo-Labeling + Large Model Ensemble

## 📊 起點（0523 結束）

- **目前 LB**：0.64596（rank 38/60）
- **目標**：突破 0.68，進入前 15-25 名
- **天花板**：合法方法 ~0.72；0.85+ 區疑似 leak（不走）

## 🚨 排行榜情勢

| 區間 | 排名 | 解讀 |
|---|---|---|
| 0.85+ | 1-7 | 疑似外部 labels leak（不正常） |
| 0.71-0.79 | 8-11 | sophisticated 合法方法 |
| 0.66-0.70 | 12-23 | 標準 BERT + 1-2 tricks |
| 0.65-0.66 | 24-37 | 標準 BERT + noweight |
| **0.64-0.65** | **38-47** | **我們在這 + baseline** |
| 0.62-0.64 | 48-50 | TF-IDF 等弱方法 |

要進 top 20，**至少**要 LB ~0.67；top 15 要 ~0.69。

## 🎯 四個突破方向（按 EV 排序）

### 🥇 方向 1：Pseudo-Labeling（核心，~70 分鐘）

**動機**：590 筆 test 跟 train 文本重複（41%），對這些用 final_d 的高信心預測作為 pseudo-label 加進 train，相當於「擴大訓練集」。對 850 筆 OOD test 也可挑高信心的當 PL。

**做法**：
1. 用 `final_d` 對 test 1444 筆做預測，記錄每筆的最高機率
2. 挑選高信心（max_prob ≥ 0.85 或 top 700 筆）作為 pseudo-label
3. 把這些 (test_text, pseudo_label) 加進 train（變成 ~13,700 筆）
4. 重訓 PubMedBERT noweight 5-fold
5. 加進 ensemble

**合法性**：使用自己模型的預測作 PL 是 Kaggle 標準合法做法。沒用 test ground truth。

**預期增益**：+0.005-0.015 LB

### 🥈 方向 2：PubMedBERT-large noweight 多 seed（~3 小時，但配額重）

我們只有 seed=42 一個 large。加 seed=2024, 7：
- 訓練：每 seed 約 25 分鐘（large 模型 epochs=3, batch=16）
- 3 seed × 5 fold = ~75 分鐘
- 加入 ensemble 後預期 +0.003-0.010 LB

### 🥉 方向 3：Test-Time Augmentation (TTA)（~30 分鐘）

對每個 test row 做 3-5 種文本變化：
- 原始
- 隨機刪除 10% 句子
- 段落順序 shuffle
- Title only
- Body only

對每種變化用 final_d 跑預測，平均所有變化的 prob。

**預期增益**：+0.002-0.008 LB

### 🏅 方向 4：DeBERTa-v3-LARGE noweight（~90 分鐘）

DeBERTa-v3-large 350M 參數，是 base (~140M) 的 2.5 倍：
- 設定：epochs=3, batch=8 (grad accum=4 → effective 32), lr=8e-6
- 預期單模型 OOF 0.67-0.70（base 是 0.645）
- 加入 ensemble +0.003-0.008 LB

## 📅 0524 執行順序

| 時段 | 動作 | 預計時間 |
|---|---|---|
| 09:00 | Cell 0: 環境 + 還原 0523 全部 runs | 3 分鐘 |
| 09:05 | **方向 1 Step A**: 用 final_d 對 test 預測 + 挑 pseudo-labels | 5 分鐘 |
| 09:10 | **方向 1 Step B**: 訓練 PubMedBERT noweight + pseudo-labels 5-fold | 60 分鐘 |
| 10:10 | **方向 1 ensemble**: final_d 4 + PL 模型 → v23 | 1 分鐘 |
| 10:15 | **方向 3 TTA**: 對 final_d + v23 做 TTA → v24 | 30 分鐘 |
| 10:45 | **第一次提交：v23 (PL ensemble)** | – |
| 10:50 | **方向 4 訓練**: DeBERTa-v3-large noweight 5-fold | 90 分鐘 |
| 12:20 | **方向 2 訓練**: PubMedBERT-large seed=2024 | 25 分鐘 |
| 12:45 | 全部 ensemble → v25 (PL + DeBERTa-large + multi-large-seed) | 1 分鐘 |
| 13:00 | **第二次提交：v24 (TTA)** | – |
| 13:30 | **第三次提交：v25 (full final)** | – |
| 視 LB | 第 4、5 次提交視情況決定 | – |

## 🎲 提交策略

5 次配額按「**確定性 × 期望 LB**」分配：

| # | 提交 | 預期 LB | 動機 |
|---|---|---|---|
| 1 | v23 (PL ensemble) | 0.65-0.67 | 確認 PL 方向有效 |
| 2 | v24 (TTA on final_d) | 0.64-0.66 | 確認 TTA 有效 |
| 3 | v25 (PL + DeBERTa-large + multi-seed) | 0.67-0.70 | 主打 |
| 4 | 保留 | – | 視前 3 個結果 |
| 5 | 保留 | – | 視前 3 個結果 |

## ⚠️ 風險控管

1. **Pseudo-label 信心門檻**：太低（0.7）會引入噪音；太高（0.99）樣本太少。**從 0.85 開始試**
2. **DeBERTa-v3-large 不穩定**：先跑 fold 0 確認，OOF < 0.55 立刻調 LR
3. **Colab A100 配額**：估全部訓練 ~4 小時。若配額不足，**先做方向 1 + 3**（合計 100 分鐘）
4. **每階段 Drive 備份**：避免 session 中斷重做

## 🎯 突破性 KPI

- **必達**：LB ≥ 0.66（rank 30 之前）
- **目標**：LB ≥ 0.68（rank 15-20）
- **stretch**：LB ≥ 0.70（rank 10-15）

如果方向 1 PL 確認有效 (LB ≥ 0.66)，全力衝方向 4 (DeBERTa-large)。
如果方向 1 沒效，回頭穩固 final_d。

## 📂 需要的新檔案

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `src/make_pseudo_labels.py` | 根據 final_d 預測，產生 pseudo-labeled train CSV（支援 `--top-n-per-class` 平衡選取）| ✅ 已完成 |
| `src/train_bert_pseudo.py` | 載入擴充 train（含 PL），訓練 5-fold | ✅ 已完成 |
| `src/tta_predict.py` | 對既有模型做 5 種 augmentation 的 TTA inference | ✅ 已完成 |
| `src/ensemble_predict.py` | 新增 `--prefer-tta` 旗標支援 TTA-augmented probs | ✅ 已更新 |

## 🆕 0524 進度（即時更新）

### EXP-022: Pseudo-Labeling — 完成但無明顯效益
- PL 單模型 OOF 0.6556（+0.003 vs noweight）
- v23 ensemble OOF 0.6586（比 final_d 0.6592 低 0.0006）
- 不提交 v23 — 節省配額

### EXP-023: DeBERTa-v3-large noweight — 進行中
- fold 0 OOF = 0.6648（19 分鐘）
- fold 1-4 訓練中（預計 76 分鐘）
- 預估 5-fold OOF: 0.650-0.660

### EXP-024: TTA — 腳本就緒
- 5 種 augmentation：original / drop_sentence / shuffle_body / truncate_head / truncate_tail
- DeBERTa-large fold 0-4 訓完後即可套用
- 預期增益 +0.002-0.008 LB

## 🎯 修正後的提交計劃

| # | 提交 | 預期 LB | 等待 |
|---|---|---|---|
| 1 | v24 = final_d + DeBERTa-large | 0.650-0.658 | DeBERTa-large 訓完 |
| 2 | v25 = v24 with DeBERTa-large TTA | 0.652-0.660 | TTA 跑完 |
| 3 | 視情況：v26 = v24 + 大模型 multi-seed | – | 看時間 |
| 4 | 保留 | – | 視結果 |
| 5 | 保留 | – | 視結果 |
