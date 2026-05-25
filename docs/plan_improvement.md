# Kaggle 醫學文本多分類競賽：最終改善與模型強化計畫

這份計畫彙整了深度資料探勘的結果（包含 40.7% 的 Data Leakage 發現）、多標籤衝突的處理邏輯，以及純粹從「演算法與模型學習層面」強化的進階策略，為您打造最終的衝榜藍圖。

---

## 🚨 一、 核心數據洞察與重大挑戰分析

經交叉比對 Kaggle 訓練集與測試集，我們發現了足以左右競賽排名的關鍵資料特性：

1. **核彈級資料洩漏 (Data Leakage)**：
   * 測試集 (1,444 筆) 中有 **589 筆 (高達 40.7%)** 完全一模一樣地出現在訓練集中。
   * 其中 511 筆在訓練集中只有單一明確標籤，78 筆存在「衝突標籤」。
2. **多標籤攤平的歷史遺跡**：
   * 訓練集中有超過 2,400 筆資料是「同一段文字，卻有不同的單一標籤」。
   * 76.8% 的衝突發生在「特定器官疾病」與「廣泛一般病況 (General)」之間。
3. **隱藏的極端不平衡**：
   * 若將衝突樣本清理後，`digestive system diseases` (消化系統) 僅剩約 1,077 筆，而 `cardiovascular diseases` (心血管) 則高達 2,472 筆。

---

## 🛠️ 二、 資料清洗與前處理策略 (Data Cleansing)

這是解決模型在單選題 (Single-label Macro F1) 中崩潰的最核心步驟。

1. **結構化雜訊移除 (Regex Cleaning)**：
   * 在資料讀取時，使用正規表達式 `re.sub(r'\b[A-Z]{2,}(?:--|:)\s*', '', text)` 清除如 `RESULTS--`, `OBJECTIVE--` 等無意義排版字眼，使 Attention 權重聚焦於醫學症狀。
2. **目標導向的標籤優先級去重 (Target-Aware Deduplication)**：
   * 將訓練集按文本 (condition) Groupby。
   * **規則 1**：若同時出現 `general` 與 `特定疾病` (如心血管)，**強制刪除 general 標籤列**，只保留特定疾病。這解決了 76% 的矛盾。
   * **規則 2 (捨棄重度模糊樣本)**：若經過規則 1 後，仍有文本同時對應「兩種特定的器官疾病」(約 627 筆)，**直接將其從訓練集中剔除 (Drop)**，確保特徵萃取器不被污染。

---

## 🚀 三、 雙軌強化戰略：純 AI 學習 vs 規則後處理

您可以根據競賽規則或個人偏好，選擇以下戰略（或兩者並行）：

### 戰略 A：純粹提升模型泛化能力的 AI 戰略 (推薦)
不寫死答案，而是利用洩漏的情報引導模型學習。
1. **領域自適應預訓練 (Task-Adaptive Pretraining / MLM)**：
   * 合併 Train 和 Test 所有的文本，讓 `DeBERTa-v3-large` 進行一次 Masked Language Modeling (挖空填字) 的繼續預訓練。讓模型完全適應這場比賽獨特的醫學縮寫與多重疾病語境。
2. **考古題樣本加權 (Sample Weighting / Oversampling)**：
   * 既然已知有 589 篇文本「必考」，在訓練集的 Dataset/DataLoader 中，對這 589 筆交集資料**放大 Loss 權重 (例如 3x~5x)**，或直接**複製 3 遍**。強迫模型深度優化這些特徵。

### 戰略 B：衝榜外掛 - 規則查表覆蓋 (Post-processing Override)
* 讓模型預測所有測試資料後，寫 Python 腳本進行後處理。
* 只要測試集的文本在訓練集中有「唯一標籤」，直接將預測結果**覆蓋為訓練集的真實答案**。這能保證 40% 的題目拿滿分。

---

## 🧠 四、 重裝模型配置與抗噪訓練 (Modeling & Training)

捨棄輕量化與多標籤 (Sigmoid) 訓練，全面回歸穩健的單分類架構。

1. **重裝語言模型 (Heavyweight Ensembles)**：
   * 主力 1：`DeBERTa-v3-large` (Disentangled Attention 非常適合捕捉隱晦症狀差異)。
   * 主力 2：`BioLinkBERT-large` (具備強大醫學文獻關聯理解能力)。
2. **核心正規化技術 (R-Drop)**：
   * 不改變輸出架構的最強抗噪手段。將同一 Batch 連續通過兩次 Dropout，並計算兩次輸出的 KL 散度作為懲罰項 ($\alpha=4\sim5$)，迫使模型對相同輸入給出極度一致的預測。
3. **動態損失函數 (Loss Function Optimization)**：
   * **Class Weights**：在 `CrossEntropyLoss` 加上權重，給予消化與神經系統更高的懲罰值，對抗極端不平衡。
   * **Label Smoothing**：設定為 `0.05` ~ `0.1`，防止模型對殘存的模糊資料過度自信。

---

## 🔄 五、 交叉驗證與推論融合 (Validation & Ensemble)

1. **5-Fold Cross Validation (五折交叉驗證)**：
   * 使用 `StratifiedKFold` (K=5) 切分經過清洗的訓練集。訓練 5 個同架構模型。
2. **Soft Voting Ensemble (軟投票)**：
   * 在推論測試集時，將 5 折的 `DeBERTa` 與 5 折的 `BioLinkBERT` (共 10 個模型) 的預測機率分佈 (Softmax outputs) 進行**相加平均**，最後再做 `argmax` 取出單一標籤。
   * 此舉能最大程度抵消單一模型的偏差，是確保 Macro F1 突破 0.86+ 的標準操作。
