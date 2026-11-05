# 作業三：多輸出學習 (Multi-output Learning)
**2025 國立清華大學 自然語言處理 (NTHU Natural Language Processing)**  
**授課教師**：高宏宇 教授 (Hung-Yu Kao)  
**助教群**：智慧型知識管理實驗室 (IKM Lab TAs)

---

## 目錄
1. [多輸出學習簡介 (Multi-output Learning)](#1-多輸出學習簡介)
2. [作業內容說明 (Assignment Description)](#2-作業內容說明)
3. [資料集介紹 (SemEval 2014 Task 1)](#3-資料集介紹)
4. [程式碼實作指南 (Code Guide & TODOs)](#4-程式碼實作指南)
   - [環境套件匯入](#41-環境套件匯入)
   - [資料載入與預處理](#42-資料載入與預處理)
   - [TODO 1：使用 PyTorch DataLoader 建立批次資料](#todo-1建立批次資料-dataloader)
   - [TODO 2：建構模型架構 (BERT-based)](#todo-2建構模型架構-multilabelmodel)
   - [TODO 3：定義優化器與損失函數](#todo-3定義優化器與損失函數)
   - [TODO 4：編寫訓練迴圈 (Training Loop)](#todo-4編寫訓練迴圈)
   - [TODO 5：驗證模型表現 (Validation Loop)](#todo-5驗證模型表現)
   - [TODO 6：測試模型表現 (Test Loop)](#todo-6測試模型表現)
5. [評分標準 (Scoring)](#5-評分標準)
6. [作業繳交規範 (Delivery Policies)](#6-作業繳交規範)
   - [檔案格式與命名規範](#61-檔案格式與命名規範)
   - [報告應包含之環境資訊](#62-報告應包含之環境資訊)
   - [程式撰寫與 AI 使用規範](#63-程式撰寫與-ai-使用規範)
   - [扣分標準 (Punishments)](#64-扣分標準-punishments)
   - [繳交平台與期限](#65-繳交平台與期限)

---

## 1. 多輸出學習簡介

### 多輸出學習 vs 多任務學習
根據論文參考文獻：
> Xu, Donna, et al. "Survey on multi-output learning." *IEEE Transactions on Neural Networks and Learning Systems* 31.7 (2019): 2409-2429.

* **多輸出學習 (Multi-output Learning)**：
  - **資料**：同一份輸入資料 $\text{data}$。
  - **標籤**：每個資料點同時具有多組標籤（$\text{labels}_1, \text{labels}_2$）。
  - **目標**：模型同時優化多個目標任務（$\text{Objective}_1, \text{Objective}_2$）。
* **多任務學習 (Multi-task Learning)**：
  - **資料**：來自不同來源或任務的資料（$\text{data}_1, \text{data}_2$）。
  - **標籤**：各任務有各自對應的標籤（$\text{labels}_1, \text{labels}_2$）。
  - **目標**：模型在不同任務與資料間共享表示並優化各自目標（$\text{Objective}_1, \text{Objective}_2$）。

---

## 2. 作業內容說明
* 在本次作業三中，你將練習建立一個**多輸出學習 (Multi-output Learning)** 模型。
* 本作業使用 **SemEval 2014 Task 1** 資料集，該資料集中的每一筆樣本皆同時對應多個標籤。

---

## 3. 資料集介紹
本作業採用 **SemEval 2014 Task 1: Evaluation of Compositional Distributional Semantic Models on Full Sentences through Semantic Relatedness and Textual Entailment** 資料集。

### 3.1 樣本範例與欄位說明
每一筆資料包含：
1. `premise`（前提句）：文字字串
2. `hypothesis`（假設句）：文字字串
3. `relatedness_score`（相關度分數）：**子任務 1（迴歸任務 Regression）**，數值區間為 $1.0 \sim 5.0$
4. `entailment_judgement`（蘊涵關係判斷）：**子任務 2（三分類任務 3-class Classification）**
   - `'0'`: 中立 (NEUTRAL)
   - `'1'`: 蘊涵 (ENTAILMENT)
   - `'2'`: 矛盾 (CONTRADICTION)

#### 範例：
| premise | hypothesis | relatedness_score (迴歸 1~5) | entailment_judgement (分類 0/1/2) |
| :--- | :--- | :--- | :--- |
| A group of kids is playing in a yard and an old man is standing in the background | A group of boys in a yard is playing and a man is standing in the background | `4.5` | `0` (NEUTRAL) |
| A man, a woman and two girls are walking on the beach | A group of people is on a beach | `4.300000190734863` | `1` (ENTAILMENT) |

### 3.2 資料集劃分數量
* **訓練集 (Train split)**：4,500 筆
* **驗證集 (Validation split)**：500 筆
* **測試集 (Test split)**：4,927 筆

---

## 4. 程式碼實作指南

### 4.1 環境套件匯入
```python
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# 如有需要，可以自行安裝與匯入其他函式庫
```

### 4.2 資料載入與預處理
> ⚠️ **重要注意事項**：需安裝 `datasets==2.21.0` 才能從 Hugging Face 正常下載 `sem_eval_2014_task_1`！

```python
class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 將中文標點符號替換為英文標點符號，避免被 Tokenizer 切為 [UNK]
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")
```
*註：快取路徑 (`cache_dir`) 可依個人需求自由設定。*

---

### TODO 1：建立批次資料 (DataLoader)
* **目標**：實作 `collate_fn` 函式並定義訓練、驗證、測試的 DataLoader。
* **說明**：
  - `collate_fn` 定義一個 batch 的資料該如何打包。
  - 輸入參數為 batch 資料（tuple/list），函式需利用 Tokenizer 將 premise 與 hypothesis 進行編碼並轉換成 Tensor，並將對應標籤打包。
  - 需回傳各子任務的資料批次與標籤。

```python
# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    pass

# TODO1-2: Define your DataLoader
dl_train = # Write your code here
dl_validation = # Write your code here
dl_test = # Write your code here
```
* **驗證輸出**：
  執行 `print(next(iter(dl_train)))` 應取得：
  1. `input_text` (`input_ids`, `token_type_ids`, `attention_mask`)
  2. `labels1`（迴歸標籤，relatedness_score）
  3. `labels2`（分類標籤，entailment_judgement）

---

### TODO 2：建構模型架構 (MultiLabelModel)
* **規定模型**：作業硬性規定基底模型必須使用 `google-bert/bert-base-uncased`。
* **禁止事項**：**不允許**使用額外的預訓練語言模型。
* **自訂彈性**：除了基底 BERT 模型之外，可以自由設計額外的神經網路架構（例如加入線性層 Linear layers、激活函數 Activation functions 或其他神經網路元件）。
* **架構示意**：
  $$\text{Input text} \longrightarrow \text{BERT} \longrightarrow \begin{cases} \text{Linear\_1} \longrightarrow \text{relatedness\_score (迴歸)} \\ \text{Linear\_2} \longrightarrow \text{entailment\_judgement (三分類)} \end{cases}$$

```python
# TODO02: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Write your code here
        # Define what modules you will use in the model
        # Please use "google-bert/bert-base-uncased" model
        # Besides the base model, you may design additional architectures by incorporating linear layers, activation functions, or other neural components.
        # Remark: The use of any additional pretrained language models is not permitted.
        
    def forward(self, **kwargs):
        # Write your code here
        # Forward pass
        pass
```

---

### TODO 3：定義優化器與損失函數
* **優化器**：建議使用 `Adam` 或 `AdamW`。
* **損失函數**：觀察兩個子任務的不同類型，需分別為不同任務設計對應的損失函數（例如：迴歸使用 MSE/L1/Smooth L1；分類使用 CrossEntropyLoss）。
* **評估指標函式**：
  - 迴歸任務（皮爾森相關係數）：`psr = load("pearsonr")`
  - 分類任務（準確率）：`acc = load("accuracy")`

```python
model = MultiLabelModel().to(device)

# TODO03-1: Define your Optimizer
optimizer = # Write your code here

# TODO03-2: Define your loss functions (you should have two)
# Write your code here

# scoring functions
psr = load("pearsonr")
acc = load("accuracy")
```

---

### TODO 4：編寫訓練迴圈
* **嚴格限制**：**本次作業不能使用 `HuggingFace Trainer`**，必須自行使用原生 PyTorch 梯度下降進行訓練。
* **流程要求**：
  1. 每個 training step 計算梯度並更新模型參數。
  2. 總損失值（Total Loss）必須是所有子任務損失的聚合（aggregation，如加權和或相加）。
  3. 標準流程：清空梯度 (`zero_grad`) $\to$ 前向傳播 (forward) $\to$ 計算 Loss $\to$ 反向傳播 (`backward`) $\to$ 模型參數優化 (`step`)。

```python
best_score = 0.0
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO04: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
```

---

### TODO 5：驗證模型表現
* 在每個 Epoch 結束後，於驗證集 (`Validation set`) 上評估模型表現。
* 輸出所有子任務的評估分數：
  - `relatedness_score`：計算 **皮爾森相關係數 (Pearson correlation coefficients)**。
  - `entailment_judgement`：計算 **準確率 (Accuracy)**。
* 保存最佳模型權重（當 `pearson_corr + accuracy > best` 時，將模型權重儲存至 `./saved_models/best_model.ckpt`）。

```python
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO05: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (PearsonCorr, Accuracy)
    pearson_corr = # Write your code here
    accuracy = # Write your code here
    
    if pearson_corr + accuracy > best:
        best = pearson_corr + accuracy
        torch.save(model.state_dict(), f'./saved_models/best_model.ckpt')
```

---

### TODO 6：測試模型表現
* 載入驗證集表現最佳的模型權重 (`best_model.ckpt`)，並在測試集 (`Test set`) 上進行推論評估。
* 計算測試集上的皮爾森相關係數與準確率。
* 此測試表現佔總成績的 10%。

```python
# Load the model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(f"./saved_models/best_model.ckpt", weights_only=True))

# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO06: Write the test loop
# Write your code here
# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation
```

---

## 5. 評分標準 (Scoring)

總分共分為三大區塊：**程式實作 (40%)**、**模型測試成效 (10%)**、**書面報告 (50%)**。

### 5.1 程式實作評分 (Coding work: 40%)
| 項目代號 | 評分項目 | 佔比 | 備註說明 |
| :--- | :--- | :---: | :--- |
| **TODO 1** | 使用 PyTorch DataLoader 建立批次資料 | 5% | 包含 `collate_fn` 與三個 DataLoader 的定義 |
| **TODO 2** | 建構模型架構 (MultiLabelModel) | 5% | **繳交的程式碼必須使用 `bert-base-uncased`，否則此 5% 不予計分** |
| **TODO 3** | 定義優化器與損失函數 | 5% | 需定義兩個任務各自對應的損失函數 |
| **TODO 4** | 編寫訓練迴圈 (Training Loop) | 5% | 原生 PyTorch 訓練迴圈，多任務 Loss 聚合 |
| **TODO 5** | 驗證模型表現 (Validation Loop) | 10% | 計算 Pearson 與 Accuracy，保存最佳 checkpoint |
| **TODO 6** | 測試模型表現 (Test Loop) | 10% | 載入最佳權重於測試集評估 |

### 5.2 模型效能分數 (Score: 10%)
* **準確率越高，得分越高**。
* **報告最後必須附上測試日誌 (Testing log) 與準確率的截圖**（僅限一張圖）。
* **基準線 (Baseline)**：
  - 在測試集上達到 **Pearson $\ge$ 0.8** 且 **Accuracy $\ge$ 0.8**，可保底獲得 **4%**。

### 5.3 書面報告評分 (Report: 50%)
報告需嚴格使用官方提供的範本檔 `NLP_HW3_report_template.docx` 撰寫：

1. **RoBERTa-base 模型實驗與比較 (15%)**：
   - 在相同資料集上訓練 `RoBERTa-base` 模型，並回報 `BERT-base` 與 `RoBERTa-base` 的效能表現。**(10%)**
   - 探討 BERT 與 RoBERTa 的主要差異，並解釋這些架構與訓練差異如何影響模型效能。**(5%)**
2. **GPT-2 模型實驗與比較 (15%)**：
   - 在相同資料集上訓練 `GPT-2` 模型並評估其效能。**(10%)**
   - BERT 與 GPT-2 的主要差異為何？深入討論兩者的主要架構差異，並分析其如何影響模型表現。**(5%)**
3. **多輸出學習 vs 單獨任務訓練 (5%)**：
   - 比較多輸出學習模型與分別為各子任務單獨訓練的兩個 BERT 模型的表現。
   - 多輸出學習是否有帶來效能提升？詳細解釋原因。**(5%)**
4. **錯誤分析與模型改善 (10%)**：
   - 為什麼模型無法正確預測某些資料點？請提供錯誤分析 (Error Analysis)，並介紹你如何嘗試改善模型效能。**(10%)**
5. **報告加分項 (5%)**：
   - 任何能使你的報告更完整、更深入的補充分析或嘗試。**(5%)**

---

## 6. 作業繳交規範 (Delivery Policies)

### 6.1 檔案格式與命名規範
作業最終需壓縮成一個 `.zip` 壓縮檔上傳，壓縮檔內僅能包含以下 3 個檔案（**嚴禁使用其他副檔名**）：

| 繳交內容 | 檔案格式 | 命名規則 | 範例 (以清華大學學號 12345678 為例) |
| :--- | :--- | :--- | :--- |
| **程式碼** | Python 檔 (`.py`) | `NLP_HW3_<學校>_<學號>.py` | `NLP_HW3_NTHU_12345678.py` |
| **書面報告** | Word 檔 (`.docx`) | `NLP_HW3_<學校>_<學號>.docx` | `NLP_HW3_NTHU_12345678.docx` |
| **套件清單** | 文字檔 (`.txt`) | `requirements.txt` | `requirements.txt` |
| **最終壓縮檔**| ZIP 壓縮檔 (`.zip`) | `NLP_HW3_<學校>_<學號>.zip` | `NLP_HW3_NTHU_12345678.zip` |

> 💡 **提示**：若在 Google Colab 上撰寫，可透過 `檔案 (File)` $\to$ `下載 (Download)` $\to$ `下載 .py (Download .py)` 匯出 Python 程式碼。

### 6.2 報告應包含之環境資訊
報告範本開頭表格必須清楚填寫執行環境資訊：
* **Colab 或 Kaggle 使用者**：
  - 執行環境：`Colab` 或 `Kaggle`
  - Python 版本：填寫當前版本（如 `3.10.12`）
  - GPU 資訊：填寫分配到的 GPU（例如從 Colab 執行 `!nvidia-smi` 查看，如 `Tesla T4`）
* **本機 (Local) 使用者**：
  - 執行環境：系統 OS（如 `Ubuntu 22.04`）、CPU 型號（如 `AMD Ryzen 7-7800X3D`）
  - Python 版本：如 `Python 3.10.1`
  - GPU 資訊：如 `NVIDIA RTX 3090 * 1`

### 6.3 程式撰寫與 AI 使用規範
1. **生成式 AI / ChatGPT 使用規定**：
   - 若有使用 ChatGPT、Copilot 或任何生成式 AI 工具輔助，**必須同時在「程式碼註解」與「書面報告」中詳盡說明具體使用方式與範圍**。
   - 未註明使用生成式 AI 者，將予以扣分。
2. **網路開源程式碼參考規定**：
   - 若參考網路上的程式碼，必須提供來源網址連結。
   - 未註明參考網路來源者，將予以扣分。
3. **嚴禁抄襲 (No Plagiarism)**：
   - 禁止直接複製貼上同學的程式碼或報告。
   - 繳交重複/高度雷同之程式碼或報告，**抄襲雙方一律 0 分**！

### 6.4 扣分標準 (Punishments)
請務必嚴格檢查下列項目，每項疏漏皆有明確扣分：

| 違規項目 | 扣分分值 |
| :--- | :---: |
| 程式碼檔案命名錯誤（僅接受 `.py`） | **-5 分** |
| 報告檔案命名錯誤（僅接受 `.docx`） | **-5 分** |
| 壓縮檔命名錯誤（僅接受 `.zip`） | **-5 分** |
| 未附上 `requirements.txt` | **-5 分** |
| 報告內未附上 Python 版本等環境資訊 | **-5 分** |
| 擅自修改程式碼範本（**僅允許修改資料載入部分**） | **-5 分** |
| 擅自修改報告範本排版/結構 | **-5 分** |
| 程式碼或報告與其他同學呈現高度雷同（抄襲） | **雙方皆以 -100 分計 (0 分)** |

### 6.5 繳交平台與期限
* **繳交平台**：請上傳壓縮檔至 **NTU COOL** 課程平台。
* **作業時間**：共有 **三週** 時間可完成本項作業。
* **助教聯絡信箱**：如有任何疑問，可寄信至 `nthuikmlab@gmail.com` 洽詢助教群。
