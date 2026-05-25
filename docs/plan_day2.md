# 0523 作戰計劃 — Multi-Architecture Ensemble

## 目標

LB **0.655+**（vs 0522 最佳 final_d 0.64596），把多架構 ensemble 推到當下能力上限。

## 🔍 Deep EDA 補充發現（影響策略）

1. **Test 二極化**：41% 近重複 train、59% 完全新（max sim < 0.5）— 模型對「新樣本」的 generalization 才是真正瓶頸
2. **Class 5 是「次要標籤」**：跟其他四類共現 55–71%，test 真實比例可能 < 33%。**不要再強推 class 5**，現在 noweight 預測 20% 可能已合理
3. **多標籤本質**：12,994 列 = 10,395 unique 文本 × ~1.25 labels/text。改用 multi-label BCE 訓練可能 root-cause fix
4. **Title-only TF-IDF OOF 0.478**：弱訊號，僅作 ensemble 多樣性參考
5. **強判別詞彙**已 mapped（cancer/cirrhosis/brain/coronary 等），BERT 應已學到

## 預算

- Colab A100：~3.5 小時訓練（SciBERT 60 分 + DeBERTa 70 分 + 視情況 large 80 分 + ensemble 15 分）
- Kaggle 提交：5 次

## 進度起點（0522 結束狀態）

| 提交 | 模型組合 | OOF | LB |
|---|---|---|---|
| final_a_noweight42 | PubMedBERT noweight × 1 | 0.6524 | 0.64133 |
| final_b_pubmed_2seeds | + seed=2024 | 0.6526 | 0.64312 |
| final_c_3noweight | + BioBERT noweight | 0.6566 | 0.64448 |
| final_d_4noweight | + PubMedBERT-large | 0.6592 | **0.64596** |
| v14_5noweight | + PubMedBERT seed=7（今晚跑完）| ~0.660 | ~0.647 |

已驗證規律：OOF→LB gap 穩定 ~−0.012；多 seed 邊際遞減；class 5 是主要瓶頸。

## 訓練計劃

### Phase 1：SciBERT noweight（60 分鐘）

- 模型：`allenai/scibert_scivocab_uncased`
- 設定：seed=42, epochs=4, batch=32, lr=2e-5, class-weight=none
- 預期 fold 平均 val_macro_f1：0.63–0.65

**檢查點**：
- ✅ ≥ 0.640：健康，繼續 Phase 2
- ⚠️ 0.620–0.640：仍加進 ensemble 試
- ❌ < 0.620：跳過

### Phase 2：DeBERTa-v3-base noweight（70 分鐘）

- 模型：`microsoft/deberta-v3-base`
- 設定：seed=42, epochs=4, **batch=24, lr=1e-5**（DeBERTa 對 lr 敏感）
- 預期 fold 平均 val_macro_f1：0.63–0.67

**檢查點**：fold 0 跑完看 val_macro_f1，若 < 0.55 立即中斷重調（嘗試 lr=2e-5 或 warmup_ratio=0.06）。

### Phase 3（可選備案）：PubMedBERT-large noweight seed=2024（80 分鐘）

跑或不跑的判斷：
- 跑：Phase 1 或 Phase 2 失敗（OOF < 0.62），需要補保險
- 不跑：Phase 1 + 2 都 ≥ 0.64，已足夠多樣性

設定：seed=2024, epochs=3, batch=16, lr=1e-5, class-weight=none

## 🆕 Phase 4：Multi-label BCE 訓練（高風險高回報，~60 分鐘）

**腳本已寫好**：`src/train_bert_multilabel.py`（已在 repo）

每個 train 文本的 label = 5-dim binary vector，把該文本在 train 中觀察到的所有標籤都標為 1（單標籤文本就是 one-hot，多標籤文本是 multi-hot）。Loss 用 BCEWithLogitsLoss + `problem_type="multi_label_classification"`。

**Inference**：sigmoid → normalize（保證和為 1，跟 CE 模型可平均）→ argmax 取單標籤。

**Train target 分布**（已驗證）：
- 7,995 rows = 1 label（one-hot）
- 4,410 rows = 2 labels
- 573 rows = 3 labels
- 16 rows = 4 labels
- 平均 1.43 labels/row

**為什麼可能大幅提升**：不再強迫模型「挑一個」對多標籤文本，模型學每個類別的獨立機率，inference 取 argmax 自然偏向 most specific label。

**Colab cell**：
```python
import os
MODEL = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
SEED = 42
for fold in range(5):
    cmd = (f'python src/train_bert_multilabel.py --model {MODEL} '
           f'--fold {fold} --seed {SEED} --epochs 4 --batch-size 32 '
           f'--lr 2e-5 --max-length 512 '
           f'--tag pubmedbert_bce_seed{SEED}_fold{fold}')
    rc = os.system(cmd); assert rc == 0

import json, glob, pandas as pd
runs = sorted(glob.glob('outputs/bert_runs/pubmedbert_bce_seed42_fold*/metrics.json'))
df = pd.DataFrame([json.load(open(p)) for p in runs])
print(df[['fold', 'val_macro_f1', 'train_secs']])
print(f'\\nBCE mean OOF Macro F1: {df.val_macro_f1.mean():.4f}')

backup('after_bce')
```

**檢查點**：
- ✅ ≥ 0.660：BCE 訓練有效，必加進 ensemble
- ⚠️ 0.640–0.660：表現與 noweight 相當，加進來看 ensemble 有沒有 diversity gain
- ❌ < 0.640：BCE 設定可能有問題，跳過

## 🆕 Phase 5：Title-only 模型（低成本，~30 分鐘）

訓練 PubMedBERT noweight 但只用文章第一句（標題）。加入 ensemble 提供 diversity。

**預期單模型 OOF**：~0.55–0.58（比全文低，但跟全文模型 error pattern 不同）
**Ensemble 收益**：+0.001–0.005

## Ensemble 候選

跑完訓練後依序做：

| 候選 | 組成 | 預期 OOF |
|---|---|---|
| v15_6models | v14 + SciBERT | 0.660–0.665 |
| v16_7models | v15 + DeBERTa | 0.668–0.675 |
| v17_no_large | v16 砍掉 large | 0.665–0.672 |
| **v18_with_bce** | v17 + multi-label BCE PubMedBERT | **0.675–0.685** ★ |
| v19_with_title | v18 + title-only model | 0.677–0.687 |
| v20_no_scibert | 備案：若 SciBERT 拖累 | 視情況 |

## 預期 LB 分數

| 提交 | 樂觀 | 保守 |
|---|---|---|
| v15 (+SciBERT) | 0.653 | 0.648 |
| v16 (+DeBERTa) | 0.660 | 0.652 |
| v17 (砍 large) | 0.658 | 0.650 |
| **v18 (+BCE)** | **0.668** | **0.658** |
| v19 (+title) | 0.670 | 0.659 |

## 提交順序（5 次配額）

按「目前已有 + 訓練完成度」分情境：

**情境 A（4 個訓練全跑完）**：
| # | 提交 | 目的 |
|---|---|---|
| 1 | v18_with_bce | OOF 最高，主打 |
| 2 | v16_7models | 對照組（看 BCE 加多少） |
| 3 | v17_no_large | 看 large 在 BCE 加入後是否還有貢獻 |
| 4 | 保留 | 看 1–3 結果決定 |
| 5 | 保留 | 留給最後 |

**情境 B（只跑完 SciBERT + DeBERTa）**：
| # | 提交 | 目的 |
|---|---|---|
| 1 | v16_7models | OOF 最高 |
| 2 | v17_no_large | 看 large 貢獻 |
| 3 | v15_6models | 看 DeBERTa 貢獻 |
| 4 | 保留 | 視結果 |
| 5 | 保留 | 視結果 |

## 提交決策樹

```
if v16 LB ≥ 0.655:
    Kaggle UI 設 v16 為 Selected
    剩餘配額試 v17 / weighted variants
elif v15 LB ≥ v14 LB + 0.003:
    DeBERTa 真有效，試 v16/v17
elif 兩個新架構都沒幫助（LB ≤ v14 + 0.001）:
    認賠，final_d 或 v14 為最終
    剩餘配額試診斷（v11, biobert_only_noweight）
```

## 時程

| 時段 | 動作 |
|---|---|
| 09:00 | Colab 開機，環境 + 還原 runs |
| 09:05 | Phase 1 SciBERT 訓練（~60 分） |
| 10:05 | SciBERT 完成，備份 |
| 10:10 | Phase 2 DeBERTa 訓練（~70 分） |
| 11:25 | DeBERTa 完成，備份 |
| 11:30 | 跑 v15 / v16 / v17 ensemble |
| 11:45 | 上 GitHub 拉新的 multi-label BCE 訓練腳本 |
| 11:50 | Phase 4 BCE 訓練（~60 分） |
| 12:55 | BCE 完成，備份，跑 v18 ensemble |
| 13:10 | 提交 #1 (v18 if BCE ≥ v17 OOF, else v16) |
| 13:30 | 看 LB 決定 #2 |
| 下午 | 視結果決定 Phase 5 (title-only) 或結束 |

## 風險控管

1. **每訓練完一個模型立即輕量備份**（patterns `*.npy, *.json, *.csv` 到 Drive）
2. **DeBERTa fold 0 早停監控**：fold 0 val_macro_f1 < 0.55 立即中斷
3. **保留 final_d 或 v14 為 Kaggle Selected**，新提交確認比舊好才 swap
4. **至少留 2 個 ensemble 候選未交**，先看頭兩個結果再決定

## 收尾流程（不論結果）

1. 跑 final backup：`predictions_only_FINAL_0523.tar.gz` 上 Drive
2. 把所有 submission 下載到本機 `outputs/`
3. Kaggle UI 確認 Selected = 公開 LB 最高 + 私下覺得最穩
4. 把 0523 結果更新到 README / plan_day1.md

## 收尾範例 Kaggle Selected 策略

通常選 2 個 submission 作為 final:
- **保守選**：當天 LB 最高的（risk averse）
- **激進選**：OOF 最高的（信 OOF）

最佳實踐：兩個各選一份（保守 + 激進），最大化 private LB 期望值。
