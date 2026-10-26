=== Page 1 ===
作業二：算術作為一種語言 (Assignment 2: Arithmetic as a language)
清華大學自然語言處理 (NTHU Natural Language Processing)
高宏宇 教授 (Hung-Yu Kao)
IKM 實驗室助教群 (IKM Lab TAs)

=== Page 2 ===
作業說明 (Assignment Description)
在作業二中，你將練習訓練簡單的序列生成模型（sequence generation models）。我們將把「算術表達式」（arithmetic expressions）視為一種特殊的語言，並使用循環神經網路（RNN、LSTM）為此語言訓練序列生成模型。
在本次作業中，你將練習神經網路模型的訓練與分析，並深入探討與反思模型對算術運算的邏輯理解能力。
2

=== Page 3 ===
訓練模型流程 (Train a model)
步驟 1：準備資料集 (Prepare the dataset)
步驟 2：建構模型 (Construct the model)
步驟 3：定義優化器 (Define Optimizer)
步驟 4：定義損失函數 (Define loss function)
步驟 5：訓練模型 (Train the model)
步驟 6：評估模型 (Evaluate the model)
3

訓練迴圈核心步驟：
1. 將資料輸入至模型 (Input data to the model)
2. 計算損失 (compute loss)
3. 清除梯度 (clear gradients)
4. 計算反向傳播梯度 (compute gradients)
5. 優化更新模型參數 (optimize parameters)
6. 回到步驟 1 (back to 1.)

=== Page 4 ===
線性回歸的簡單範例 (A simple example of linear regression)
● 資料分佈 (Data distribution)：
● 使用一條直線來代表/擬合這些資料 (Use a line to represent these data)
4

=== Page 5 ===
線性回歸的 PyTorch 程式碼 (Pytorch code of linear regression)
5

# 玩具資料集 (Toy dataset)
x_train = torch.tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=torch.float32)

y_train = torch.tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=torch.float32)

# 線性回歸模型 (Linear regression model)
model = nn.Linear(input_size, output_size)

# 損失函數與優化器 (Loss and optimizer)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 訓練模型 (Train the model)
for epoch in range(num_epochs):
    # 步驟 1. 前向傳播輸入資料至模型
    outputs = model(x_train)
    # 步驟 2. 計算損失
    loss = criterion(outputs, y_train)

    # 步驟 3. 清除梯度
    optimizer.zero_grad()
    # 步驟 4. 反向傳播計算梯度
    loss.backward()
    # 步驟 5. 優化更新參數
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

對照標註：
- 初始化資料張量 (initialize data tensor)
- 定義模型 (model)
- 損失函數與優化器 (loss & optimizer)
- 載入資料 (load data)
1. 輸入資料至模型
2. 計算損失
3. 清除梯度
4. 計算梯度
5. 優化更新參數

=== Page 6 ===
前向與反向傳播深入解析 (1/2) (Forward & Back-propagation Insight 1/2)
6

步驟 1 (step 1)：
optimizer.zero_grad()
print_grads(model)

weight: Parameter containing:
tensor([[0.4165]], requires_grad=True)
weight grad: None  （清空梯度後，權重梯度為 None）

bias: Parameter containing:
tensor([0.4819], requires_grad=True)
bias grad: None  （清空梯度後，偏差梯度為 None）

步驟 2 (step 2)：
loss.backward()
print_grads(model)

weight: Parameter containing:
tensor([[0.4165]], requires_grad=True)
weight grad: tensor([[10.0239]])  （計算出權重梯度）

bias: Parameter containing:
tensor([0.4819], requires_grad=True)
bias grad: tensor([1.3666])  （計算出偏差梯度）

=== Page 7 ===
前向與反向傳播深入解析 (2/2) (Forward & Back-propagation Insight 2/2)
7

步驟 2 (step 2)：
loss.backward()
print_grads(model)

weight: Parameter containing:
tensor([[0.4165]], requires_grad=True)
weight grad: tensor([[10.0239]])

bias: Parameter containing:
tensor([0.4819], requires_grad=True)
bias grad: tensor([1.3666])

步驟 3 (step 3)：
optimizer.step()
print_grads(model)

weight: Parameter containing:
tensor([[0.4065]], requires_grad=True)  （參數值根據梯度進行更新：0.4165 -> 0.4065）
weight grad: tensor([[10.0239]])

bias: Parameter containing:
tensor([0.4805], requires_grad=True)  （參數值根據梯度進行更新：0.4819 -> 0.4805）
bias grad: tensor([1.3666])

步驟 4 (step 4)：
optimizer.zero_grad()
print_grads(model)

weight: Parameter containing:
tensor([[0.4065]], requires_grad=True)
weight grad: None  （再次清空梯度，避免累加）

bias: Parameter containing:
tensor([0.4805], requires_grad=True)
bias grad: None

=== Page 8 ===
輸出結果 (Outputs)
8

各 Epoch 訓練損失數值：
Epoch [5/60], Loss: 11.2489
Epoch [10/60], Loss: 4.6657
Epoch [15/60], Loss: 1.9987
Epoch [20/60], Loss: 0.9182
Epoch [25/60], Loss: 0.4805
Epoch [30/60], Loss: 0.3031
Epoch [35/60], Loss: 0.2313
Epoch [40/60], Loss: 0.2021
Epoch [45/60], Loss: 0.1903
Epoch [50/60], Loss: 0.1855
Epoch [55/60], Loss: 0.1835
Epoch [60/60], Loss: 0.1827

圖表：原始資料點分佈 (紅點) 與 擬合回歸直線 (藍線)

=== Page 9 ===
RNN 觀念複習 (RNN Review)
9
● 每個輸入的詞嵌入向量（word embeddings）都有一個對應的輸出 y。
● 在生成式任務中，RNN 將先前的所有 token 編碼成一個向量表示，並輸出 y 作為預測的下一個 token。

架構圖流程：
h0 -> [LSTM/RNN (h_{t-1})] -> [LSTM/RNN (h_t)] -> [LSTM/RNN (h_{t+1})]
             ^                      ^                      ^
          x_{t-1}                  x_t                  x_{t+1}
             |                      |                      |
             v                      v                      v
          y_{t-1}                  y_t                  y_{t+1}

=== Page 10 ===
算術運算任務 (Arithmetic)
10
● 你的任務是訓練一個 RNN 循環神經網路模型，使其具備執行算術運算的能力。

LSTM -> LSTM -> LSTM -> LSTM -> LSTM
  ^       ^       ^       ^       ^
 '1'     '+'     '1'     '='     '2' (紅色字元代表目標答案)
  |       |       |       |       |
  v       v       v       v       v
  y1     y1+    y1+1    y1+1=   y1+1=2

=== Page 11 ===
資料集說明 (Dataset)
算術資料集 (Arithmetic dataset)
● 訓練集大小 (Train split)：1,000,000 筆 (pieces)
● 評估集大小 (Eval split)：10,000 筆 (pieces)
● 每一筆資料樣本：由 2～3 個數字組成的等式，每個數字皆在 [0, 50) 範圍內
  ○ 例如：(10 + 4) * 2 = 且答案為 28
  ○ 運算符號包含：+, -, *, ()
11

=== Page 12 ===
資料集範例 (Dataset examples)
12
*紅色字元代表答案
● 任務形式：A (+/-/*) B (+/-/*) C = ?

| 完整範例 (Example) | 模型輸入 (Inputs) | 目標答案 (Answer) |
|---|---|---|
| 1 + 2 - 3 = 0 | 1 + 2 - 3 = | 0 |
| (10 + 4) * 2 = 28 | (10 + 4) * 2 = | 28 |

=== Page 13 ===
程式碼實作 (Code)
13

=== Page 14 ===
Colab：掛載 Google 雲端硬碟 (1/2) (Colab: access google drive 1/2)
14
在此輸入這兩行程式碼：
from google.colab import drive
drive.mount('/content/gdrive')

登入你的 Google 帳號授權存取

=== Page 15 ===
Colab：掛載 Google 雲端硬碟 (2/2) (Colab: access google drive 2/2)
15
將你的資料上傳至 Google 雲端硬碟：
- arithmetic_eval.csv
- arithmetic_train.csv
- main.ipynb
上傳至路徑：我的雲端硬碟 > NTHU_NLP_HW2

在 Colab 中的路徑對應為：(./drive/MyDrive/...)

=== Page 16 ===
檢查下載的檔案 (Check the downloaded file)
16
以逐行方式檢視 (Line-by-line manner)
Arithmetic_train.csv
包含 id、輸入算式、真實答案，以逗號「,」分隔

src,tgt
14*(43+20)=,882
(6+1)*5=,35
13+32+29=,74
31*(3-11)=,-248
24*49+1=,1177
3+(25*25)=,628
8*(30+10)=,320
9*38+49=,391
23-17=,6
23-26*15=,-367
18*22-19=,377
(47-23)*42=,1008
7-1+46=,52
(45+2)*25=,1175
47+(29-1)=,75
(27+41)-12=,56
38+(29-46)=,21
32*(19+28)=,1504
37*(35-24)=,407
24*(22-49)=,-648
(4-41)*6=,-222
24-39*6=,-210
38+(0-20)=,18
26-2-35=,-11

=== Page 17 ===
載入資料 (Load the data)
17
從 .csv 檔案讀取資料：
df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
df_train.head()

將輸出資料轉換為字串：
# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

=== Page 18 ===
TODO1: 在此建立字典 (5%) (Build your dictionary here)
18
建立字典 (Build Dictionary)：
• 模型無法直接使用純文字進行計算。
• 必須將所有文字（數字/符號）轉換為數值表示。
• 特殊 Token：
  ○ '<pad>'
    - 一個批次（Batch）內的每個句子長度可能不同。
    - 較短的句子需以 '<pad>' 進行長度補齊，以對齊該批次中最長的句子。
  ○ '<eos>'
    - 指定生成序列的結束點。
    - 若沒有 '<eos>'，模型將不知道何時停止生成。

char_to_id = {}
id_to_char = {}

# 在此撰寫你的程式碼
# 建立字典，並為訓練資料集中的每個 token 分配一個 id
# 字典必須包含 <eos> 與 <pad>
# char_to_id 是將字元轉換為 id，而 id_to_char 則是相反轉換

vocab_size = len(char_to_id)
print('Vocab size{}'.format(vocab_size))

字典範例：
char_to_id = {
  '<pad>' : 0,
  '<eos>' : 1,
  '0' : 2,
  ...
}

反向字典範例：
id_to_char = {
  0 : '<pad>',
  1 : '<eos>',
  2 : '0',
  ...
}

=== Page 19 ===
TODO2: 資料前處理 (10%) (Data preprocessing)
19
資料前處理 (Data Preprocessing)：
• 將資料處理為模型輸入與輸出所需的格式（以 <eos> 標籤結尾）。

# 在此撰寫你的程式碼
df.head()

將資料轉換為模型輸入與輸出所需的格式。
此處我們將等號前不需要預測的部分替換為 '<pad>'：

| 索引 | src | tgt | len | char_id_list | label_id_list |
|---|---|---|---|---|---|
| 0 | 0+0=0 | 0 | 5 | [15, 3, 15, 17, 15, 1] | [0, 0, 0, 0, 15, 1] |
| 1 | 0-0=0 | 0 | 5 | [15, 7, 15, 17, 15, 1] | [0, 0, 0, 0, 15, 1] |
| 2 | 0*0=0 | 0 | 5 | [15, 13, 15, 17, 15, 1] | [0, 0, 0, 0, 15, 1] |
| 3 | (0+0)*0=0 | 0 | 9 | [14, 15, 3, 15, 10, 13, 15, 17, 15, 1] | [0, 0, 0, 0, 0, 0, 0, 0, 15, 1] |
| 4 | 0+0*0=0 | 0 | 7 | [15, 3, 15, 13, 15, 17, 15, 1] | [0, 0, 0, 0, 0, 0, 15, 1] |

=== Page 20 ===
TODO3: 資料批量打包 (5%) (Data Batching)
20
資料批量打包 (Data Batching)：
• 使用 `torch.utils.data.Dataset` 建立名為 dataset 的資料生成工具。
• 接著，使用 `torch.utils.data.DataLoader` 從 dataset 中隨機抽樣，並將樣本分組打包為批次（batches）。
• 範例：1+2-3=0
  ○ 模型輸入：1 + 2 - 3 = 0
  ○ 模型標籤輸出：/ / / / / 0 <eos>  （其中的 '/' 會被替換為 <pad>）
  ○ 模型輸出的關鍵在於：模型不需要預測前半部算式的下一個字元。真正重要的是一旦模型看到 '=' 符號，就應該開始生成答案（即 '0'），在生成答案後，還必須生成 <eos>。

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        # 回傳資料總筆數
        return # 在此撰寫你的程式碼
    
    def __getitem__(self, index):
        # 從資料中提取輸入資料 x 與真實標籤 y
        x = # 在此撰寫你的程式碼
        y = # 在此撰寫你的程式碼
        return x, y

在 DataLoader 中，資料首先透過 Dataset 的 `__getitem__(...)` 方法取出以建構批次，接著傳遞至 `collate_fn` 進行填充補齊等後續處理。
模型僅需對輸入中 '=' 符號之後的 token 進行預測。
模型在 '=' 符號之前產生的任何輸出都是無關的，在訓練過程中必須排除在損失計算之外。

=== Page 21 ===
模型設計 (Model)
21
模型設計中嚴格禁止使用 Attention 機制 (Attention is not allowed in model design)。

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()
        # 在此撰寫你的程式碼
        # 設計你自己的模型，應包含 embedding 層、數個 hidden 層以及一個 linear 線性層
        
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        self.rnn_layer1 = torch.nn.RNN(input_size=embed_dim,
                                       hidden_size=hidden_dim,
                                       batch_first=True)

        self.rnn_layer2 = torch.nn.RNN(input_size=hidden_dim,
                                       hidden_size=hidden_dim,
                                       batch_first=True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

此處我們定義了兩層 RNN 層。

=== Page 22 ===
TODO4: 答案生成 (10%) (Generation)
22
def generator(self, start_char, max_len=200):
    char_list = [char_to_id[c] for c in start_char]
    next_char = None
    
    while len(char_list) < max_len:
        # 在此撰寫你的程式碼
        # 將 char_list 打包為 Tensor
        # 將 Tensor 分別送入 embedding 層、LSTM 層、linear 線性層
        y = # 取得下一個 token 的預測機率分佈 y
        next_char = # 使用 argmax 函式取得下一個預測字元的 id
        
        if next_char == char_to_id['<eos>']:
            break
        
        char_list.append(next_char)
        
    return [id_to_char[ch_id] for ch_id in char_list]

將 `start_char` 輸入模型。每次輸入序列時，模型會對下一個 token 產生預測。下一個 token 的預測結果對應至模型輸出序列中的最後一個元素。
當輸出為 '<eos>' 時，應停止生成。

=== Page 23 ===
TODO5: 模型訓練 (5%) 與 TODO6: 評估驗證 (10%)
23
● 你必須使用 Teacher Forcing（教師強制）技術訓練 LSTM 模型。
  ○ 請務必確認模型是在 GPU 上進行訓練。
● 你必須計算評估集上的準確率（Exact Match 完全匹配準確率）。
  ○ 必須使用 `generator` 函式生成完整答案，並檢查是否與真實答案（ground truths）完全吻合。

=== Page 24 ===
Teacher Forcing 教師強制技術 (Teacher forcing)
24
Teacher forcing 是一種常用於序列模型（sequence-based models）的訓練技術。
在 Teacher forcing 中，訓練期間並非使用模型自身預測的輸出作為下一步的輸入，而是直接將訓練資料中的「真實目標（ground truth）」作為下一步的輸入。

例如：1+2-3=0
模型的原始輸入為："1+2-3="
與其執行多次前向傳播來逐步自回歸生成完整序列，我們直接輸入："1+2+3=0"，並同時預測 $p(\text{'0'} \mid \text{'1+2+3='})$ 與 $p(\text{'<eos>'} \mid \text{'1+2+3=0'})$。

=== Page 25 ===
Teacher Forcing 原理示意 (Teacher forcing)
25
未使用 Teacher forcing（Without teacher forcing）：
例如：真實標籤是 "I am a student"，但模型預測輸出為 "I am eating dinner"
LSTM -> LSTM -> LSTM -> LSTM -> LSTM
'<bos>' -> 'I' -> 'am' -> 'eating' (紅字錯誤) -> 'dinner' (紅字錯誤)
y 輸出：'I' -> 'am' -> 'eating' (預測) -> 'dinner' (預測) -> '<eos>'
使用模型先前的預測值作為下一個輸入 token（容易造成錯誤累積）。

使用 Teacher forcing（Teacher forcing）：
LSTM -> LSTM -> LSTM -> LSTM -> LSTM
'<bos>' -> 'I' -> 'am' -> 'a' (真實標籤) -> 'student' (真實標籤)
y 輸出：'I' -> 'am' -> 'eating' (預測) -> 'dinner' (預測) -> '<eos>'
一律使用資料集中的真實 token 作為序列輸入。

=== Page 26 ===
作業繳交 (Submission)
26

=== Page 27 ===
評分機制：程式碼實作 (Scoring: Coding work 40%)
27
程式碼實作 (Coding work) : 40%

| 實作任務 (TODOs) | 成績佔比 (Scores) |
|---|---|
| TODO1: 在此建立字典 (Build your dictionary here) | 5% |
| TODO2: 資料前處理 (Data preprocessing) | 5% |
| TODO3: 資料批量打包 (Data Batching) | 5% |
| TODO4: 答案生成 (Generation) | 10% |
| TODO5: 模型訓練 (Training) | 5% |
| TODO6: 評估驗證 (Evaluation) | 10% |

=== Page 28 ===
評分機制：排行榜與書面報告 (Scoring)
28
程式碼實作 (Coding work) : 40%
排行榜 (Leaderboard) : 15%（達成之準確率越高，獲得分數越高）
● 助教會在未公開的測試集（private test set）上執行你提交的 checkpoint 進行評估。

書面報告 (Report) : 45%
● 描述你的模型架構、資料前處理方法與超參數設定，並論證你的選擇理由。(5%)
● 若改用 2 層 LSTM 或 GRU 來取代 RNN，答案生成的品質會發生什麼變化？為什麼？(10%)
● 若我們使用三位數構建訓練集，而評估集由兩位數構建，答案生成的品質會發生什麼變化？(10%)
● 若我們構建一個包含 20% 錯誤答案的訓練集，這將如何影響生成回應的品質？請列舉一些範例說明。(10%)
● 為什麼在訓練過程中需要梯度裁剪（gradient clipping）？(5%)
● … 任何能加強你報告內容的補充分析。(5%)

為便於助教批改，建議盡量以「文字/表格」呈現數據，而非僅提供截圖。

=== Page 29 ===
繳交規定：檔案格式 (Delivery policies: File formats)
29
● 程式碼檔案：Jupyter Notebook 檔 (`.ipynb`)
  ○ 請透過 Colab 下載你的腳本檔（附帶執行輸出）。
● 套件清單：`requirements.txt`
  ○ 例如：`numpy==1.26.3`
● 書面報告：Microsoft Word 檔 (`.docx`)
● **嚴禁使用任何其他格式**。
● 上傳作業前，請將上述檔案全部打包成 `.zip` 壓縮檔。

=== Page 30 ===
繳交規定：檔名規範 (Delivery policies: Filenames)
30

| 檔案類型 | 檔名規則 (Filename rule) | 檔名範例 (Filename example) |
|---|---|---|
| 程式碼實作 | NLP_HW2_school_student_ID.ipynb | NLP_HW2_NTHU_12345678.ipynb |
| 模型權重檔 | checkpoint.pt | checkpoint.pt |
| 套件清單 | requirements.txt | requirements.txt |
| 壓縮檔案 | NLP_HW2_school_student_ID.zip | NLP_HW2_NTHU_12345678.zip |

=== Page 31 ===
繳交規定：報告中必須包含的資訊 (Delivery policies: Things You should include)
31
● 在你的書面報告開頭必須包含：

| 環境類型 (Environment types) | 若使用 Colab 或 Kaggle | 若使用本機環境 (If local) |
|---|---|---|
| 執行環境 (Running environment) | Colab | System: Ubuntu 22.04, CPU: Ryzen 7-7800X3D |
| Python 版本 (Python version) | Colab | Python 3.10.1 |

=== Page 32 ===
繳交規定：程式撰寫與學術倫理規範 (Delivery policies: Rules of coding)
32
● 若你有使用生成式 AI（Generative AI），請在以下兩處皆明確標註使用情況：
  ○ 程式碼註解中 (Code comments)
  ○ 書面報告中 (Reports)
● 嚴禁抄襲。絕對不可複製貼上同學的作業。
● 若有參考網路上的程式碼，請務必附上參考連結來源。
● 發生以下行為將會導致作業被扣分處分：
  (1) 使用生成式 AI 未依規定註明
  (2) 引用網路資源未依規定標註
  (3) 抄襲行為

=== Page 33 ===
違規懲罰條款 (Punishments)
33

| 違規項目 (Rule) | 懲罰 / 扣分 (Punishment) |
|---|---|
| 程式檔名格式不符：NLP_HW2_school_student_ID.ipynb（僅接受 .ipynb） | -5 分 |
| 壓縮檔檔名格式不符：NLP_HW2_school_student_ID.zip | -5 分 |
| 未包含 requirements.txt 套件清單 | -5 分 |
| 未包含 checkpoint.pt，或模型中使用了 Attention 注意力機制 | Leaderboard = 0 分 |
| 書面報告中未註明 Generative AI 的使用聲明 | -10 分 |
| 書面報告中未標註 Python 版本 | -5 分 |
| 擅自修改程式碼模板（僅允許修改資料載入相關路徑） | -5 分 |
| 程式碼或報告與其他同學的繳交成果呈現高度雷同（抄襲） | 雙方皆 -100 分 |

若使用 Colab，請至：檔案 (File) → 下載 (Download) → 下載 .ipynb 以取得 Python notebook 檔案。

=== Page 34 ===
上傳壓縮檔案 (Uploading the zipped file)
34
● 請將你的 `.zip` 壓縮檔上傳至 NTU COOL。
● 本次作業共有三週的時間可以完成。
