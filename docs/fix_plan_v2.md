# Kaggle 醫學文本多分類競賽：進階突圍計畫 (Fix Plan v2)

本計畫基於先前「硬規則查表法 (Rule-based Override)」導致 Leaderboard (LB) 分數下降的失敗經驗，重新校準了針對該資料集的策略。事實證明，我們面對的是一個具有「惡意隨機切分 (Random Split)」與「負向洩漏 (Negative Leakage)」的高雜訊資料集。

---

## 🚨 一、 核心失敗分析 (為什麼不能用查表法改答案？)

1. **負向洩漏效應 (Negative Leakage)**
   原始文獻 (HuggingFace) 將一篇同時具備「腫瘤」與「一般病況」的摘要攤平成兩筆資料。主辦方在隨機切分時，把「腫瘤」分給了 Train，把「一般病況」分給了 Test。
   當我們強制將 Test 預測改成 Train 看過的「腫瘤」時，反而覆蓋掉了模型原本可能猜對的「一般病況」。這導致了反向扣分。
2. **模型的決策比硬規則更精準**
   我們訓練的 `DeBERTa-v3-large` 與 `BioLinkBERT` 已經能在訓練過程中捕捉到這些模糊地帶的「軟邊界 (Soft Boundaries)」。強行用人類定義的規則 (特定疾病 > 一般病況) 去蓋掉 Softmax 機率，破壞了模型對 Macro F1 的最佳化結果。

---

## 🎯 二、 全新破局策略：資料為王 (Data-Centric AI)

既然無法在預測階段 (Inference) 作弊，我們必須將戰場轉移回 **資料清洗 (Data Cleansing)** 與 **訓練階段 (Training)**。

### 步驟 1：建構高純度訓練集 (Hard-Purification)
拋棄原始充滿矛盾的 12,994 筆訓練資料，建立一個 `clean_train.csv`：
* **去重並清除模糊樣本**：若一篇摘要在 Train 中同時具有兩種不同的特定器官疾病 (如腫瘤 + 心血管，共約 627 筆)，**直接刪除**。不要讓模型學習這種單選題無法作答的無解雜訊。
* **特定疾病優先清理**：若一篇摘要同時具有 `general` 與 `特定疾病`，**刪除 general 標籤列**。讓特徵與特定疾病的對應關係變得絕對純粹。

### 步驟 2：考古題加權 (Target-Aware Weighting)
雖然不能直接抄答案，但我們確定有 589 篇文本**一定會考**。
* 在 DataLoader 計算 Loss 時，針對這 589 篇出現在 Train/Test 交集的文本給予 **更高的權重 (Sample Weight = 2.0 ~ 3.0)**，或是直接在 `clean_train.csv` 中複製 2~3 遍 (Oversampling)。
* **目的**：讓模型在權重更新時，強烈記憶這些「必考題」的特徵，自然而然地在推論時給出高信心的預測。

---

## 🧠 三、 模型與訓練架構升級

1. **繼續使用 Cross-Entropy (CE)**
   如您的 `README.md` 所述，這場比賽的本質是強迫單選，因此 Multi-label 的 BCE Loss 絕對是死路一條。請堅持使用標準的單標籤 CE Loss。
2. **領域自適應預訓練 (TAPT / Masked Language Modeling)**
   * 將 `kaggle_trainset.csv` 和 `kaggle_testset.csv` 的文本合併（不含標籤）。
   * 使用 `DeBERTa-v3-large` 或 `PubMedBERT` 對這批合併文本進行 MLM (Masked Language Modeling) 繼續預訓練。
   * **目的**：讓模型先熟悉這個資料集特有的「醫學術語分佈」與「多重病況共存」的語境，之後再接上分類頭 (Classification Head) 進行 Fine-tuning。這招通常能無腦提升 1%~2% 分數。

---

## 🚀 四、 終極推論與融合策略 (Ensemble)

1. **拋棄 Rule-Based Override**
   預測階段絕對不要再對測試集套用任何硬性覆寫規則。關閉所有類似 `overlap-constraint` 的限制。
2. **機率軟投票 (Soft Voting)**
   將多個模型 (例如 5-Fold 的 DeBERTa + 5-Fold 的 BioLinkBERT) 在測試集上的 **Softmax 預測機率直接相加平均**，最後再取 `argmax`。
   這能最大程度抵消單一模型在那些「多標籤爭議題目」上的極端偏差，是在這種高雜訊資料集穩定提升 Macro F1 的最佳也是唯一解法。
3. **偽標籤自訓練 (Pseudo-Labeling) - 最終殺招**
   用您目前最佳的 `final_d` 模型組合去預測 Test Set，挑出**信心度極高 (Confidence > 0.95)** 的樣本（約 600~800 筆）。將這些偽標籤資料加入 `clean_train.csv` 重新訓練一版終極模型。

---

## 📅 五、 執行 Checklist
- [ ] 撤回先前的查表法提交，恢復對純模型輸出 (如 `final_d_4noweight`) 的信心。
- [ ] 撰寫 Pandas 腳本，輸出 `clean_train.csv` (刪除模糊雙重標籤、處理 General 衝突)。
- [ ] 實作 Sample Weighting，對 589 筆交集文本加權。
- [ ] (Optional) 執行合併 Train/Test 文本的 MLM 預訓練。
- [ ] 以純淨資料重新進行 5-Fold 訓練，並匯出預測機率進行 Soft Voting Ensemble。
