# Assignment 3: Fact Verification 提交內容審查報告與「吸引子」深度剖析

> 📅 **審查時間**：2026-11-11  
> 🎯 **審查對象**：Commit `4c8ddaf` (`Complete assignment 3 notebook and study guides`) 及 Assignment 3 相關提交  
> 📂 **核心關聯檔案**：
> - [`2026/Assignments/Assignment3/st_code.ipynb`](st_code.ipynb)
> - [`2026/Assignments/Assignment3/st_question.md`](st_question.md)
> - [`2026/Assignments/Assignment3/作業規範.md`](作業規範.md)
> - [`2026/Assignments/Assignment3/程式逐段解析.md`](程式逐段解析.md)
> - [`2026/Assignments/Assignment3/index.md`](index.md)

---

## 📌 執行摘要（Executive Summary）

本次審查針對 GPT 最新提交的繁體中文事實查核作業（CFEVER）全套實作與指引文件進行「宏觀架構檢查」、「微觀代碼驗證」與「系統吸引子分析」。

* **整體評價**：代碼架構非常完整、工整，**比助教提供的原始 `sample_code.ipynb` 更健壯**（修復了 sample code 原生的空證據組假命中、候選句子只限定 train 標註句之不合理限制、GPU 記憶體未釋放等問題）。
* **直接扣分隱患**：
  1. `st_code.ipynb` 最下方的 **Report 包含大量「待填」佔位符**，若未執行實驗並填入真實數值即繳交，Report（佔比 40%）將嚴重失分。
  2. 簡報規定最終繳交檔名必須為 **`NLP_HW3_school_student_ID.ipynb`** 與 **`.zip`**，違者各扣 5 分。
* **核心效能瓶頸（吸引子）**：
  1. `wiki_pages` 候選頁面池僅取自 `train.jsonl`，為 Dev 與 Kaggle Test 設立了**召回率的硬上限（Ceiling）**。
  2. 預設 `TRAIN_SAMPLES = 3000` 與 `EPOCHS = 1` 僅適用於快速走通流程，直接預測會導致 Kaggle 嚴重欠擬合。

---

## 第一部分：宏觀檢查（Macro-Level Audit）

### 1. 課程規範與助教最新限制合規性

| 規範項目 | 助教公告與答疑要求 | 當前實作狀況 | 合規判定 | 說明與改進建議 |
| :--- | :--- | :--- | :---: | :--- |
| **模型參數上限** | 單一模型參數量需小於 **1B** | 使用 `google-bert/bert-base-chinese`（約 1.02 億參數） | ✅ **完全合規** | 檢索與分類皆加入參數斷言檢查，合規無虞。 |
| **外部 API 限制** | **禁止**呼叫外部推論 API（如 OpenAI / Claude API） | 全地端 / Colab PyTorch 執行，無 API 調用 | ✅ **完全合規** | 模型權重下載自 Hugging Face 官方 Repo，屬合法權重獲取。 |
| **資料集範圍** | 僅允許使用 **CFEVER** 資料集 | 僅使用 `train.jsonl`, `dev.jsonl`, `test.jsonl`, `wiki_pages.jsonl` | ✅ **完全合規** | 未混入未授權之外部私有資料集。 |
| **Report 格式** | **嚴禁更動** Report Markdown 格式 | 嚴格保留 5 大題標題與 `Answer:` 開頭 | ⚠️ **需注意** | 格式無誤，但回答中含有**未填寫的實驗佔位符**。 |
| **繳交命名** | `NLP_HW3_school_student_ID.ipynb` 及 `.zip` | 當前分支檔案為 `st_code.ipynb` | ⚠️ **需注意** | 下載繳交前務必手動替換為正式校名與學號。 |

### 2. 管線流向（Pipeline Flow）
整體架構採用標準三階段解耦流程：
```
[ Claim 輸入 ]
      │
      ▼
【Stage 1: Document Retrieval】
  - 候選頁面：wiki_pages (取自 train 標註頁)
  - 模型：凍結的 bert-base-chinese (Mean Pooling + L2 Norm)
  - 排序：Cosine Similarity 矩陣運算 ──► Top-5 頁面
      │
      ▼
【Stage 2: Evidence Retrieval】
  - 候選句子：Top-5 頁面內所有非空句子 (保留維基原始 sentence_id)
  - 模型：同上凍結 BERT
  - 排序：Cosine Similarity ──► Top-5 句子座標與串接 Context
      │
      ▼
【Stage 3: Claim Verification】
  - 輸入：[CLS] Claim [SEP] Evidence Context [SEP]
  - 模型：bert-base-chinese + SequenceClassificationHead (3 類別)
  - 優化器：AdamW (lr=2e-5, clip_grad_norm=1.0)
      │
      ▼
【Output: submission.csv & metrics.json】
  - 驗證集輸出本地 Strict Score、Accuracy、Recall@5
  - 測試集生成標準 Kaggle 格式 CSV (NEI 自動清空為 `[]`)
```

### 3. 跨環境執行相容性
* **Google Colab**：支援 Google Drive 自動掛載與 ZIP 唯讀安全解壓，避免 Windows/Linux 路徑編碼問題。
* **Kaggle Notebook**：自動探測 `/kaggle/input` 與 `/kaggle/working`，支援無網路環境下的離線模型載入。
* **本地環境**：預設讀取當前目錄與 `bert_student_output`。

---

## 第二部分：微觀代碼與張量分析（Micro-Level Analysis）

逐 Cell 深入檢查程式邏輯、張量算子維度與邊界條件：

### 1. 資料解析與邊界防禦（Cell 9）
* **Sample Code 原生缺陷**：原始 sample code 處理 `evidence` 時未防禦空列表，若 `group = []`，`set().issubset(...)` 恆為 `True`，會導致空證據誤判為命中。
* **GPT 改進**：撰寫 `gold_groups()` 函式，嚴格校驗每個 pair 必須滿足 `len(item) >= 4`、`isinstance(item[2], str)` 且 `sentence_id >= 0`，有效過濾髒資料。

### 2. BERT 向量編碼之數學正確性（Cell 11, 15）
* **Mean Pooling**：
  $$\mathbf{u} = \frac{\sum_{i=1}^{L} \mathbf{h}_i \cdot m_i}{\max(\sum_{i=1}^{L} m_i, 1)}$$
  代碼中藉由 `mask = inputs['attention_mask'].unsqueeze(-1).float()` 精確排除 Padding token 的干擾，並以 `.clamp(min=1)` 杜絕除以零風險。
* **$L_2$ Normalization & Cosine Similarity**：
  $$\mathbf{v} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} \implies \text{Sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}^\top$$
  向量均經 `F.normalize(..., p=2, dim=1)` 歸一化，使得後續矩陣乘法 `@` 精確等價於餘弦相似度。加入 `query_batch_size=256` 分批運算，有效防止數萬筆資料同時相乘導致 OOM。

### 3. 句子檢索與索引映射（Cell 19, 20）
* **重大邏輯修正**：
  - 助教原始代碼：在提取句子時加上了 `(title, sentence_id) in wanted_sentences`。這意味著候選句子被限定在「訓練集標記過的句子」，Dev/Test 的新句子根本檢索不到！
  - GPT 代碼：將 Stage 1 取回的頁面（`selected_pages`）中**所有非空句子**納入候選庫，符合真實檢索情境。
* **二維索引回溯安全**：
  建立 `page_to_sentence_indices` 字典，先在區域候選句上進行 `scores.topk`，再以 `index = candidate_indices[local_index]` 映射回全局座標 `(title, sentence_id)`，座標對齊無誤。

### 4. GPU 記憶體主動釋放（Cell 19）
* 在句子編碼完成後，執行：
  ```python
  retrieval_model.to('cpu')
  del retrieval_model
  torch.cuda.empty_cache()
  ```
  為後續 Stage 3 載入分類模型釋放顯存，防止 Colab T4 (16GB) 發生崩潰。

### 5. 分類微調與 Kaggle 格式校驗（Cell 24, 28, 36）
* **標籤容錯映射**：使用 `.str.upper().map(label_to_id)`，大小寫統一轉換，杜絕標籤不匹配產生的 `NaN`。
* **梯度保護**：微調時加入 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`。
* **輸出嚴格檢查**：
  - NEI 標籤之證據強制格式化為 `"[]"`。
  - 使用 `assert len(submission) == len(test_df)` 與 `assert submission['id'].tolist() == test_df['id'].tolist()` 保證送交格式不變形。

---

## 第三部分：五大「吸引子（Attractor Traps）」深度剖析

「吸引子」指系統中容易讓模型、流程或開發者不自覺陷入**局部最優、指標虛高或隱性失敗**的結構性偏差。

```
                      [ CFEVER Fact Verification Pipeline ]
                                       │
      ┌────────────────────────────────┼────────────────────────────────┐
      ▼                                ▼                                ▼
【吸引子 1：候選池天花板】          【吸引子 2：抽樣幻覺】             【吸引子 3：級聯誤差累積】
wiki_pages 只讀 train 標註頁      TRAIN=3000, DEV=300            未微調檢索引發噪聲訓練
(Dev/Test 召回率鎖死硬上限)      (跑通速度快，但競賽嚴重弱化)       (錯誤 Context + 正確標籤)
      │                                │                                │
      └────────────────────────────────┼────────────────────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
           【吸引子 4：NEI 捷徑學習】           【吸引子 5：繳交格式自動扣分】
          空證據/低相似度預測 NEI              Report 待填佔位符 / 檔名錯誤
```

### 🕳️ 吸引子 1：候選頁面池截斷天花板（Candidate Pool Ceiling Attractor）—— **核心瓶頸**
* **機制**：Cell 9 中：
  ```python
  for row in all_train_df.itertuples():
      # 僅從 train 收集 wanted_pages
  wiki_pages = {page['id']: page for line in file if page['id'] in wanted_pages}
  ```
* **陷阱剖析**：
  候選維基百科僅載入 `train.jsonl` 出現過的頁面。如果 `dev.jsonl` 或 Kaggle 評分之 `test.jsonl` 的 Claim 需要的佐證頁面未曾在訓練集出現，**Stage 1 檢索率直接為 0**。
* **影響**：這為整個系統的 Recall 和 FEVER Score 設下了一道人為的天花板（即 GPT 程式印出的 `document_candidate_coverage`）。若想在排行榜取得高分，必須突破此限制（例如改用 BM25 檢索全量維基庫）。

### 🕳️ 吸引子 2：採樣縮減的「跑通幻覺」吸引子（Subsampling Illusion Attractor）
* **機制**：Cell 3 設定了 `TRAIN_SAMPLES = 3000`（全量 24,012 筆）、`DEV_SAMPLES = 300`（全量 3,000 筆）、`EPOCHS = 1`。
* **陷阱剖析**：
  這大幅提升了測試與 Debug 速度（數分鐘即可完成），但也容易讓人產生「已經完成」的錯覺。直接使用 3000 筆資料與 1 個 epoch 產生的模型對 6000 筆 test 進行預測，模型將嚴重欠擬合，Kaggle 排行榜（佔總分 30%）將嚴重失分。

### 🕳️ 吸引子 3：級聯誤差累積與噪聲訓練吸引子（Cascade Noise Injection Attractor）
* **機制**：
  Stage 3 訓練分類器時，使用的 `train_contexts` 是由未經微調的 BERT 檢索出來的句子，而非 Ground Truth 標準答案。
* **陷阱剖析**：
  當前期檢索出錯時，不相關的句子會與標籤（Supports / Refutes）強行配對，將檢索誤差轉化為分類器的**標籤噪聲（Label Noise）**，干擾模型對事實因果關係的判斷。

### 🕳️ 吸引子 4：NEI 類別的「捷徑學習」吸引子（Spurious Feature Shortcut Attractor）
* **機制**：當檢索模組無法找到高相似度句子時，串接的 Context 為空或極短。
* **陷阱剖析**：
  模型極易學到捷徑——「只要 Context 為空或相似度低，就直接預測 `NOT ENOUGH INFO`」。這雖然在指標上有基本分數，但模型並未真正學會分辨「證據已足夠反駁（Refutes）」與「真正缺乏資訊（NEI）」。

### 🕳️ 吸引子 5：繳交格式自動審查吸引子（Submission Metadata Trap）
* **機制**：
  Report 中的回答文字架構完備、文筆通順，極易讓人忽略其中的**「待填」佔位符**。
* **陷阱剖析**：
  自動批改腳本或助教人工審核時，若發現 Q1～Q3 的數值為「待執行後依第 5 節輸出填入數值...」，將直接判定為報告未完成，扣減報告配分（40%）。

---

## 第四部分：具體行動指引路線圖（Actionable Roadmap）

```
[ 當前狀態: 程式碼完備 ] 
       │
       ▼
【Step 1: 冒煙測試 (Smoke Test)】
  - 保持目前 TRAIN_SAMPLES=3000, DEV_SAMPLES=300
  - 在 Colab 執行 Run All，確認 Drive 掛載、解壓與管線跑通無誤
       │
       ▼
【Step 2: 正式衝榜 (Full Training)】
  - 調整 Cell 3 參數：
    * TRAIN_SAMPLES = None  (使用全量 24,012 筆)
    * DEV_SAMPLES = None    (使用全量 3,000 筆)
    * EPOCHS = 2 ~ 3
  - 重新執行全流程，生成正式的 submission.csv
       │
       ▼
【Step 3: 填寫正式 Report】
  - 複製 Cell 34 終端印出的真實數據 (Recall@5, Accuracy, Strict Score)
  - 填入 Report 第 1, 2, 3, 4 題的 Answer 中
  - 將 submission.csv 上傳至 Kaggle，獲取 Public Score 並填入 Report 開頭
       │
       ▼
【Step 4: 打包繳交】
  - 將 notebook 重命名為 `NLP_HW3_學校_學號.ipynb`
  - 壓縮為 `NLP_HW3_學校_學號.zip` 並上傳至 NTU COOL
```

---

*審查結論：本份實作之代碼邏輯與工程結構具備優秀水準，請協作者依據上述 Roadmap 解除「吸引子 1 & 2」，填入實測數據後即可完成高質量繳交。*
