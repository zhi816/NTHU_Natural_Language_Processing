# NLP HW1: Python 程式實作百寶箱 ＆ NLP 核心學術理論複習手冊

> **本手冊定位**：
> 1. **【前半部：Python 程式實作、套件與函式語法大全】**：為 Python 基礎與實務量身打造，將作業中所有出現過的**套件（Gensim, NumPy, Pandas, NLTK, Sklearn, Matplotlib）、內建方法（.strip, .split, .startswith, zip, set 等）、陣列切片、語法細節與經典除錯（Debug）**集中統整，便於未來隨時查閱與複習。
> 2. **【後半部：NLP 核心學術理論、詞向量幾何學與實證分析】**：聚焦於自然語言處理核心概念（分佈假說、類比幾何、靜態 vs 動態嵌入、CBOW vs Skip-Gram、語料消融實證等）。
> 
> 全篇維持你最熟悉的問答格式（學生提問與核心解析），結構清晰，杜絕零碎雜訊。

---

# 💻 第一部：Python 程式實作、套件與常用函式語法大全

## 一、 Python 內建字串核心方法（String Methods）

### Q1: `.strip()` 方法的作用與必要性
* **學生提問**：「`.strip()` 到底是在做什麼？為什麼讀取檔案時每一行都要加 `.strip()`？」
* **核心解析**：
  1. **功能**：移除字串**「開頭與結尾」**的所有空白字元（包含空格 `' '`、換行符號 `\n`、回車符號 `\r`、Tab 鍵 `\t`）。
  2. **作業中的必要性**：
     * 當我們用 `for line in f:` 逐行讀取文字檔時，Python 讀進來的每一行結尾都自帶一個隱形的換行符號 `\n`（例如 `"Athens Greece Baghdad Iraq\n"`）。
     * 若不加 `.strip()`：
       * 字串長度會多算 1 位元。
       * 判斷開頭（如 `line.startswith(":")`）時，若前面有空格就會判斷失準。
       * 透過 `.strip()` 清洗後，能確保拿到的是乾淨的純單字文字。

---

### Q2: `.split()`、`.splitlines()` 與 `.strip()` 的區別與混用
* **學生提問**：「`.split()`、`.splitlines()` 跟 `.strip()` 差在哪裡？作業裡為什麼用 `.split()`？」
* **核心解析**：
  1. **`.strip()`**：**只除雜質，不切片**。把頭尾的空格與換行拿掉，回傳**「一個清洗後的字串」**。
  2. **`.split()`**：**字串切片大師**。
     * **預設行為 `line.split()`**：以「任意連續空白字元（空格、Tab、換行）」為分隔線，將字串切碎並打包成**「字串清單（List of Strings）」**。
     * 範例：`"Athens Greece Baghdad Iraq".split()` → `['Athens', 'Greece', 'Baghdad', 'Iraq']`。
     * 在類比推論時，我們就是靠 `.split()` 一口氣解包出 4 個單字（word_a, word_b, word_c, word_d）。
  3. **`.splitlines()`**：**專門依換行切行**。只會在遇到 `\n` 或 `\r\n` 時切斷，等同於以行（Line）為單位拆解長篇文章。

---

### Q3: `.startswith()` 方法與大小寫轉換方法（`.lower()` / `.upper()`）
* **學生提問**：「`startswith` 怎麼用？為什麼曾出現 `AttributeError: 'str' object has no attribute 'startwith'`？」
* **核心解析**：
  1. **正確拼寫為 `.startswith()`**（注意有 **s**，動詞第三人稱單數）：
     * 作用：檢查字串是否以指定的子字串開頭，回傳布林值（`True` 或 `False`）。
     * 作業實作：`if line.startswith(":")`，用來瞬間捕捉章節標題行（例如 `: capital-common-countries`）。
     * 若少打 `s` 寫成 `line.startwith(":")`，Python 在 `str` 類別找不到該方法，就會噴出 `AttributeError`。
  2. **大小寫轉換方法**：
     * **`word.lower()`**：全轉小寫（例如 `"Iraq".lower()` → `"iraq"`）。
       * **重要性**：Word2Vec 與 GloVe 預訓練模型詞表全為小寫（Uncased）。如果答案是 `"iraq"` 但你的預測產出 `"Iraq"`，字串比對時 `"Iraq" == "iraq"` 會被判為 `False`，導致準確率直接歸零！因此比對前兩者必須統一 `.lower()`。
     * **`word.upper()`**：全轉大寫（例如 `"ntou".upper()` → `"NTOU"`）。

---

## 二、 Python 核心資料結構與進階迴圈語法

### Q4: 列表（List）、集合（Set）去重與字典（Dict）的選擇
* **學生提問**：「在 t-SNE 收集單字時，為什麼先用 List，再用 `set()`，最後又轉回 `list()`？」
* **核心解析**：
  1. **資料結構特性對比**：
     * **`list`（列表）**：有順序、**允許重複**、以索引 `[0]` 取值。
     * **`set`（集合）**：無順序、**絕對不允許重複**、查找速度極快（O(1) 雜湊表）。
     * **`dict`（字典）**：鍵值對（Key-Value），以鍵查值（如 `acc_20_cat['Semantic']`）。
  2. **`set()` 去重三部曲**：
     * 題庫中 `: family` 有 506 題，每題 4 個字，總共有 506 × 4 = 2,024 個字，但絕大多數都是重複出現的（如 father, mother, king, queen 重複了幾百次）。
     * `temp_words`（List）收集了 2,024 個單字。
     * `set(temp_words)`：瞬間過濾掉 1,978 個重複項目，只剩下 **46 個不重複的核心單字**。
     * `list(set_words)`：因為後續 t-SNE 需要固定單字順序並用索引提取向量，所以再轉回 List 使用。

---

### Q5: 巢狀迴圈推導式（Nested List Comprehension）
* **學生提問**：「`sentences = [[w.lower() for w in sent if w.isalpha()] for sent in gutenberg.sents()]` 這段程式碼是什麼意思？」
* **核心解析**：
  這是 Python 最經典、最高效的**雙層巢狀列表推導式**，由外而內拆解如下：
  1. **外層迴圈**：`for sent in gutenberg.sents()` → 逐句取出古典文學中的每一個句子（每個句子本身是一個單字清單）。
  2. **內層迴圈**：`for w in sent` → 遍歷這句話裡面的每一個單字。
  3. **過濾條件**：`if w.isalpha()` → 檢查該字元是否純為英文字母（自動剃除標點符號如逗號 `,`、句號 `.`、分號 `;`、引號等）。
  4. **元素轉換**：`w.lower()` → 將過濾後的英文字全數轉為小寫。
  * **等價的一般寫法**：
    ```python
    sentences = []
    for sent in gutenberg.sents():
        clean_sent = []
        for w in sent:
            if w.isalpha():
                clean_sent.append(w.lower())
        sentences.append(clean_sent)
    ```
    推導式寫法不僅程式碼簡潔（只有一行），而且在 CPython 直譯器底層經過 C 語言級別優化，執行速度比一般 `append` 迴圈快約 20%~30%。

---

### Q6: 內建函式 `zip()` 與 `enumerate()` 的配對技巧
* **學生提問**：「繪圖時寫的 `for word, (x, y) in zip(words, vectors_tsne):` 是什麼意思？」
* **核心解析**：
  1. **`zip()`（拉鍊函式）**：
     * 將多個長度相同的可迭代物件（如清單），像拉鍊一樣一對一咬合打包。
     * 範例：
       * `words = ['king', 'queen']`
       * `vectors_tsne = [[1.2, 3.4], [-0.5, 2.1]]`
       * `zip(words, vectors_tsne)` 會產出：`('king', [1.2, 3.4])` 與 `('queen', [-0.5, 2.1])`。
     * 配合解包語法 `for word, (x, y) in ...`，一次迴圈就能同時拿到單字名字 `word`、2D 平面橫坐標 `x` 與縱坐標 `y`，讓標註文字程式碼變得極端優雅。
  2. **`enumerate()`（列舉計數器）**：
     * 遍歷清單時同時產出「索引編號」與「元素內容」：
       ```python
       for rank, (word, score) in enumerate(sims, 1):
           print(f"第 {rank} 名：{word} (相似度: {score:.3f})")
       ```

---

## 三、 NumPy 矩陣運算與多維陣列切片（NumPy Essentials）

### Q7: `np.array()` 是什麼？為什麼不直接用原生 Python List？
* **學生提問**：「為什麼提取詞向量後，一定要 `np.array(vectors)`？直接用 Python 原生的 List 有什麼缺點？」
* **核心解析**：
  1. **底層記憶體連續性（Contiguous Memory）**：
     * **Python 原生 List**：本質是「記憶體指針陣列（Array of Pointers）」。List 裡存放的不是數值本身，而是分散在記憶體各處的浮點數物件地址。存取時 CPU 必須不斷隨機跳轉尋找記憶體，產生大量快取未命中（Cache Miss）。
     * **NumPy Array**：在記憶體中配置**一整塊完全連續的二進位位元組區塊**。
  2. **SIMD 向量化指令並行加速**：
     * 連續儲存允許現代 CPU 啟用 **AVX/SSE 指令集**，一條指令同時計算多組浮點數點積，運算速度比 Python 原生迴圈快 10 到 100 倍。
  3. **演算法套件強制規範**：
     * Scikit-learn 的 `TSNE`、`PCA` 等底層呼叫 C/Fortran 編譯的高效線性代數庫（BLAS/LAPACK），強制要求輸入必須具備 `(n_samples, n_features)` 矩陣結構的連續記憶體指針，原生 List 無法直接支援。

---

### Q8: NumPy 二維矩陣切片語法 `[:, 0]` 與布林遮罩（Boolean Masking）
* **學生提問**：「`vectors_tsne[:, 0]` 的冒號與逗號是什麼意思？還有 `data[mask]` 是怎麼篩選的？」
* **核心解析**：
  1. **二維矩陣切片語法 `[Row, Column]`**：
     * `vectors_tsne` 是一個形狀為 `(46, 2)` 的二維 NumPy 陣列（46 個單字，每個單字降維後有 X, Y 兩個坐標）。
     * **逗號 `,`**：用來分隔維度（逗號前面代表「列 / Row」，逗號後面代表「欄 / Column」）。
     * **冒號 `:`**：代表**「全選（Slice all）」**。
     * **`vectors_tsne[:, 0]`**：選取所有列（46 個單字），但只要第 0 欄（即所有的 **X 橫坐標**）。
     * **`vectors_tsne[:, 1]`**：選取所有列，但只要第 1 欄（即所有的 **Y 縱坐標**）。
  2. **布林遮罩（Boolean Masking）**：
     * `mask = data["SubCategory"] == ": family"` → 產生一個長度為 19,544 的布林值陣列（全是 `True` 或 `False`）。
     * `data[mask]` → 告訴 Pandas/NumPy 只保留遮罩中對應為 `True` 的那幾列，這是高階資料分析中最核心、最高效的條件過濾手段。

---

## 四、 Pandas 資料分析核心操作（Pandas DataFrame）

### Q9: 狀態機資料清洗（TODO1）與建立 DataFrame
* **學生提問**：「TODO1 是如何把雜亂的 `questions-words.txt` 轉為 DataFrame 表格的？」
* **核心解析**：
  1. **狀態機設計模式（State Machine Pattern）**：
     * 文字檔中混雜著「以冒號開頭的章節分類行（如 `: capital-world`）」與「一般題目行（如 `Athens Greece Baghdad Iraq`）」。
     * 設定兩個**狀態變數**：`current_category = ""`、`current_sub_category = ""`。
     * 逐行掃描：
       * 遇到 `:` 開頭 → 更新當前章節狀態變數。
       * 遇到一般文字 → 記錄該題目，並自動綁定當前狀態變數的值。
  2. **字典打包轉換為 DataFrame**：
     ```python
     df = pd.DataFrame({
         "Question": questions,
         "Category": categories,
         "SubCategory": sub_categories
     })
     ```
  3. **儲存 CSV 規範**：
     * 呼叫 `df.to_csv("questions-words.csv", index=False)`。
     * **注意 `index=False`**：如果不加，Pandas 會自動在第一欄寫入 0, 1, 2... 的列索引，下次讀檔時會多出一欄無效的 `Unnamed: 0`。

---

### Q10: 經典錯誤除錯：`TypeError: list indices must be integers or slices, not str`
* **學生提問**：「程式碼噴出 `TypeError: list indices must be integers or slices, not str` 是什麼原因？如何解決？」
* **核心解析**：
  1. **報錯原因**：
     * 程式碼試圖執行：`data["Question"]` 或 `data["Category"]`。
     * 這種 `變數["欄位名稱"]` 的語法**只有在變數是 `pd.DataFrame`（表格）時才合法**。
     * 如果先前的儲存格曾執行過 `data = open(...).readlines()`，導致變數 `data` 在記憶體中被覆蓋成一般的 Python `list`。對清單使用字串取值 `list["Question"]` 就會拋出此錯誤（清單索引只能是整數如 `list[0]`）。
  2. **標準防呆解法（Type Guard）**：
     ```python
     if "data" not in locals() or not isinstance(data, pd.DataFrame):
         data = pd.read_csv("questions-words.csv")
     ```
     這段程式碼會主動檢查記憶體中的 `data` 是否為合法的 DataFrame，若否則自動重新讀檔，避免執行順序錯亂導致變數型態被污染。

---

## 五、 Gensim Word2Vec 核心 API 與串流工程

### Q11: `most_similar` 參數、回傳資料結構與 `[0][0]` 的原理
* **學生提問**：「`model.wv.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]` 後面的 `[0][0]` 是什麼意思？」
* **核心解析**：
  1. **回傳資料結構**：
     * `most_similar` 預設回傳的是一個**「包著 Tuple 的 List」**，格式為：
       `[("預測單字", 餘弦相似度分數)]`
  2. **雙重索引拆解**：
     * 假設回傳 `[("iraq", 0.6812)]`：
     * **第 1 個 `[0]`**：取出清單中的第 0 個候選項目（取出元組 `("iraq", 0.6812)`）。
     * **第 2 個 `[0]`**：取出該元組中的第 0 個元素（取出字串 `"iraq"`）。
     * 如果要取相似度分數，則是寫 `[0][1]`（取出 `0.6812`）。
     * 透過 `[0][0]` 即可精準提取出純文字單字，作為預測答案。

---

### Q12: 巨量文字串流載入器 `LineSentence` 原理與記憶體防爆
* **學生提問**：「訓練 Word2Vec 時為什麼必須用 `sentences = LineSentence("wiki_texts_20.txt")`？」
* **核心解析**：
  1. **傳統讀取的致命缺點**：
     * 若寫 `sentences = [line.split() for line in open("wiki_texts_20.txt")]`，會一次性在記憶體中建立數千萬個小清單與字串物件，這會直接吃爆 Colab 的 12GB/25GB RAM，觸發作業系統強制終止程式（OOM Crash）。
  2. **`LineSentence` 的磁碟串流機制**：
     * `LineSentence` 是 Gensim 專為大文字設計的**延遲載入迭代器（Lazy-loading Iterator）**。
     * 它只在 Word2Vec 訓練迴圈需要時，才從硬碟中讀取下一行並分詞，產出後隨即釋放。
     * 它內部自動處理了 `seek(0)` 檔案指針重置，因此能完美支援 `epochs=5` 的多次重複輪巡訓練，同時記憶體消耗維持在幾 MB 的極小常數級別。

---

### Q13: 模型儲存、載入與 OOV（未登錄詞）異常防護
* **學生提問**：「模型如何保存？做類比預測時為什麼要包 `try...except`？」
* **核心解析**：
  1. **模型持久化（Persistence）**：
     * 儲存：`my_model.save("my_model.model")`（可同步儲存至 Google Drive）。
     * 載入：`my_model = Word2Vec.load("my_model.model")`。
  2. **OOV（Out-Of-Vocabulary）防護**：
     * 題庫中若有生僻字被 `min_count=5` 過濾掉，呼叫 `most_similar` 會直接引發 `KeyError`。
     * **健壯寫法**：
       ```python
       try:
           guess = model.wv.most_similar(positive=[w[1], w[2]], negative=[w[0]], topn=1)[0][0]
           preds.append(guess.lower())
       except:
           preds.append("")  # OOV 填入空字串，視為答錯，保證迴圈不中斷
       ```

---

## 六、 Scikit-learn 與 Matplotlib 降維繪圖實戰

### Q14: t-SNE 降維實作流程與主要參數
* **學生提問**：「`TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(vectors)` 每一項代表什麼？」
* **核心解析**：
  1. **`n_components = 2`**：
     * 目標降維度數。我們要畫在 2D 平面散點圖上，因此降為 2 維坐標 (x, y)。
  2. **`random_state = 42`**：
     * 隨機數種子。鎖定 t-SNE 的隨機梯度下降起點，確保每次點擊執行時，散點圖的形狀、旋轉角度與排版完全一致，不會隨機翻轉。
  3. **`perplexity = 10`**：
     * 困惑度，粗略對應考量的有效鄰居數量。在 46 個家庭關係單字的小樣本中，設為 10 能避免局部幾何關係被全域拉扯撕裂。
  4. **`.fit_transform(vectors)`**：
     * Scikit-learn 的統一介面：先「學習流形特徵（Fit）」，再「輸出降維坐標（Transform）」，輸入為 `(46, 100)` 的高維矩陣，輸出為 `(46, 2)` 的平面坐標陣列。

---

### Q15: Matplotlib 散點圖繪製與文字標註（`plt.annotate`）
* **學生提問**：「如何把單字標籤畫在散點圖的點旁邊？」
* **核心解析**：
  1. **繪製散點**：
     ```python
     plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
     ```
  2. **逐點標註文字（`plt.annotate`）**：
     ```python
     for word, (x, y) in zip(words, vectors_tsne):
         plt.annotate(word, (x, y), fontsize=7)
     ```
     * `word`：要顯示的字串（如 `"king"`）。
     * `(x, y)`：該單字在圖上的坐標位置。
     * `fontsize=7`：適當縮小字級，避免單字之間互相重疊遮擋。
  3. **儲存圖片**：
     ```python
     plt.savefig("word_relationships.png", bbox_inches="tight")
     ```
     * `bbox_inches="tight"`：自動剪裁圖片邊緣的多餘留白，確保邊界文字不被切除。

---

## 七、 NLTK 自然語言處理工具箱實務

### Q16: NLTK 經典名著語料載入與分詞模型依賴（`punkt` / `punkt_tab`）
* **學生提問**：「調用 NLTK 的 Gutenberg 語料庫時，程式碼該怎麼寫？為什麼要 download punkt？」
* **核心解析**：
  1. **完整標準載入範例**：
     ```python
     import nltk
     nltk.download('gutenberg', quiet=True)
     nltk.download('punkt', quiet=True)
     nltk.download('punkt_tab', quiet=True)
     from nltk.corpus import gutenberg

     # 取得已經切好句子的語料清單
     sentences = [[w.lower() for w in sent if w.isalpha()] for sent in gutenberg.sents()]
     ```
  2. **`gutenberg.sents()` 運作機制**：
     * 它已預先將 18 部經典文學（莎士比亞、簡·奧斯汀、聖經等）切分為 98,552 個句子。
  3. **為什麼必須下載 `punkt` 與 `punkt_tab`？**：
     * 英文中的句點 `.` 兼具「縮寫」與「句末」雙重意義。`punkt` 是無監督句子切分模型，能精準識別縮寫避免切錯行；`punkt_tab` 則是適配 Python 3.11+ 現代環境的表格化更新檔。

---

# 📖 第二部：NLP 核心學術理論、詞向量幾何學與實證分析

### Q17: 分佈假說（Distributional Hypothesis）哲學起源與反義詞悖論
* **學生提問**：「分佈假說的核心哲學是什麼？它為何導致反義詞高度相似？」
* **核心解析**：
  1. **核心名言**：J.R. Firth (1957) —— *"You shall know a word by the company it keeps."*（觀其伴，知其意）。
  2. **理論精髓**：詞彙的語意由它在大量文本中所出現的「上下文共現分佈」所決定。Word2Vec 將此過程轉化為向量空間點積最大化。
  3. **反義詞悖論（Antonym Paradox）**：
     * 英語語法中，反義詞對（如 `good` 與 `bad`）通常具有完全同構的句法槽位。
     * 因為共享幾乎相同的上下文分佈，Word2Vec 會將它們拉扯到過於相近的空間位置（實測中 `good` 與 `bad` 相似度高達 0.748），這是純上下文靜態模型無法單靠資料規模克服的先天弱點。

---

### Q18: 靜態詞向量之本質瓶頸（一詞多義崩塌）與動態詞嵌入（BERT）之演進
* **學生提問**：「Word2Vec 與現代 Transformer/BERT 最大的差別是什麼？」
* **核心解析**：
  1. **一詞多義空間崩塌（Polysemy Collapse）**：
     * 靜態詞向量（Word2Vec / GloVe）強行為每個單字指派**單一固定向量**。當單字兼具多重語意時（如 `apple` 水果與科技公司），靜態向量只能在多種語境之間取折衷加權，無法動態切換。
  2. **動態情境化詞嵌入（Contextualized Embeddings）**：
     * 現代 Transformer（如 BERT）引入自注意力機制（Self-Attention），根據句子即時計算權重。同一個單字在不同句子中會獲得完全不同的向量，徹底解決了一詞多義的混淆問題。

---

### Q19: 為什麼 Word2Vec 不適合 GPU 加速？計算物理特性深度解析
* **學生提問**：「為什麼深度學習常用 GPU，但 Word2Vec 用 GPU 反而更慢？」
* **核心解析**：
  1. **GPU 強項**：大規模**密集矩陣乘法（Dense Matrix Multiplications）**，如 Transformer 的注意力權重並行點積。
  2. **Word2Vec 特性**：極端**稀疏的查表與微調（Sparse Lookup & Update）**。每次滑動窗口只抓取幾個字與負樣本微調權重。
  3. **PCIe 傳輸瓶頸**：將單詞從 CPU 傳至 GPU 的 I/O 通訊延遲遠高於計算本身。因此 Word2Vec 跑在多核心 CPU（搭配 High-RAM）上吞吐量最高。

---

### Q20: 跨領域語料實證（Wikipedia vs Gutenberg）：時代斷層與消融極限
* **學生提問**：「為什麼 Gutenberg 古典文學訓練出的模型，類比準確率暴跌至 1.85%？調參能改善嗎？」
* **核心解析**：
  1. **時代與主題斷層（Chronological Gap）**：
     * 類比任務題庫依賴現代世界實體常識（如各國首都與現代貨幣）。
     * 18~19 世紀古典文學中完全不存在歐元、現代國家疆界等概念，實體關鍵詞全為未登錄詞（OOV），導致 `: capital`、`: currency` 幾近全部歸零（0.00%）。
  2. **消融實驗結論（Ablation Limit）**：
     * 實測將詞頻過濾門檻放寬（`min_count=5` → `min_count=2`），雖然詞表增加了 27%，但現代地緣實體依然全為 0.00%。
     * **學術結論**：**「語料庫的時代與主題斷層，是無法透過超參數調校來逆轉的物理極限」**。知識覆蓋度才是決定模型上限的關鍵。
