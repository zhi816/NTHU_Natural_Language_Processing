# NLP HW3 學習問答筆記 (st_question.md)

> 對應程式：[st_minecode.ipynb](st_minecode.ipynb)（v1 正式作業）、[st_minecode_study.ipynb](st_minecode_study.ipynb)（v1 練功本）、[st_minecode_v2.ipynb](st_minecode_v2.ipynb)（v2）、[st_minecode_study_v2.ipynb](st_minecode_study_v2.ipynb)（v2 練功本）。入口與執行步驟看 [index.md](index.md)。v2 的問題從 Q44 開始。
> 這份是照我 HW2 的發問習慣，先把這次「大概會問的問題」整理好。下面的小例子都是示意，**不是實際跑出來的分數**。

---

## 📅 2026-11-11：開工前 — 這次到底要做什麼？為什麼這樣做？

### Q1. 這次作業一句話講完是什麼？

給一句 claim，系統要：① 從維基找出相關頁面（Document Retrieval）→ ② 從頁面挑出證據句（Evidence Retrieval）→ ③ 判斷 `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO`（Classification）。

排名用 **FEVER score**：標籤要對，而且 SUPPORTS / REFUTES 的**前 5 句證據要完整找到某一組標準證據**才算分；NEI 只看標籤。

```text
claim「周杰倫在 2003 年發行了專輯葉惠美」
  ① 文件：['周杰倫', '葉惠美', '2003年', ...]
  ② 證據：[['葉惠美', 0], ['周杰倫', 5], ...]   ← 頁名 + 原始句號
  ③ 標籤：supports
```

---

### Q2. Claude 和 Gemini 都給了提分建議，到底哪個比較好？最後為什麼這樣選？

兩邊的**大方向都對**：分數卡在檢索，不是分類器；要擴大到完整維基、用標題比對、用專門的檢索模型和 reranker。差別在於「能不能在 Kaggle 30 小時 / Colab 免費 GPU 裡真的跑完」和「有沒有違反助教最新規定」。

| 建議 | 誰提的 | 我的判斷 | 最後有沒有用 |
| :--- | :--- | :--- | :--- |
| 擴大到完整維基、候選句不限 train 標註 | 兩邊都有 | ✅ 效益最大，一定要做 | ✅ |
| 標題比對（claim 裡的實體 = 頁名） | 兩邊都有 | ✅ 便宜又有效，不用 GPU | ✅ |
| BM25（`rank_bm25` / jieba） | 兩邊都有 | ❌ 一百多萬頁用純 Python 的 `rank_bm25`，每個 claim 都要和每一頁算分，跑不完；jieba 對繁中也不準 | ❌ 改用「標題比對 + 向量」 |
| 換掉 bert-base-chinese 做檢索 | 兩邊都有 | ✅ 原生 BERT 沒做過檢索訓練 | ✅ bge-m3 |
| cross-encoder reranker | 兩邊都有 | ✅ 但只能用在少量候選上 | ✅ 只重排前 20 句 |
| 分類器用 large 模型 | 兩邊都有 | 🟡 會再好一點，但訓練時間約 3 倍 | ❌ 先用 base，報告寫成「之後可以做」 |
| 訓練時混入標準證據、NEI 用檢索結果 | 兩邊都有 | ✅ 和 FEVER score 的規則最搭 | ✅ |
| 固定給滿 5 句、`DEV_SAMPLES = None` | Claude | ✅ 不花成本 | ✅ |
| NEI logit 加 bias，在 dev 調 | Claude | ✅ 幾秒鐘的事，還能變成報告第 4 題 | ✅ |
| **外部 NLI 資料集**（OCNLI / CMNLI） | Gemini | ❌ **助教 11/11 說只能用 CFEVER dataset**，Gemini 引用的是 11/06 的舊答覆 | ❌ |
| 2～3 個 large 模型 ensemble | Gemini | ❌ GPU 時數翻好幾倍，換來 1～2 分 | ❌ |
| 沿超連結做多跳檢索 | Gemini | ❌ 太複雜，而且不確定 CFEVER 的 `lines` 有沒有連結 | ❌ |
| 「+15～20%」這類預估數字 | Gemini | ⚠️ 沒有根據，不要寫進報告 | — |

另外 Claude 那段說「`st_code.ipynb` 只是原樣複製」是錯的：`st_code.ipynb` 是 GPT 改過的版本，原樣複製的是舊的 `st_minecode.ipynb`。

**結論**：取兩邊共同的核心（完整維基 + 標題比對 + 專門的檢索模型 + reranker + 補進標準證據），拿掉跑不完的（全維基 BM25、ensemble）和有違規風險的（外部資料集）。

---

### Q3. 這版和 GPT 做的 `st_code.ipynb` 差在哪？

GPT 版本基本上是 sample code 加上很多防呆，**方法沒變**：候選頁還是只有 train 標註頁、檢索還是 bert-base-chinese mean pooling、分類器還是 3000 筆 1 epoch，報告留了一堆「待執行後填入」。

| | GPT `st_code.ipynb` | 這版 `st_minecode.ipynb` |
| :--- | :--- | :--- |
| 候選頁 | train 標註頁（有天花板） | 完整維基標題比對 + train 標註頁 |
| 檢索模型 | bert-base-chinese mean pooling | bge-m3（[CLS]）+ bge-reranker |
| 分類訓練 | 3000 筆、1 epoch、檢索結果 | 全部 train、2 epochs、補進標準證據 |
| 報告 | 一段一段「待填」 | 第 7 節自動印出填好數字的完整報告，還有 5 張表 |

---

### Q4. 最後要交 Kaggle，為什麼不分成 Kaggle 版和 Colab 版兩份？

因為兩個平台的差別只在「資料在哪、輸出存哪」，大約 15 行。其他 95% 的程式一模一樣。

- 第 1 節第二格會自動判斷：有 `/kaggle/input` 就是 Kaggle、`import google.colab` 成功就是 Colab，其他就是本機。
- 分成兩份的話，之後改一個 bug 要改兩次，很容易一份改了一份忘了，報告數字也可能對不起來。
- **NTU COOL 只收一份 `.ipynb`**，本來就只能交一份。

Kaggle 和 Colab 的實際操作步驟寫在 [index.md](index.md)。

---

### Q5. 用 bge-m3、bge-reranker 這些別人訓練好的模型，算不算違反「只能用 CFEVER dataset」？

不算。規定分兩件事：

1. **資料集**：助教 11/11 說「只能使用 cfever dataset」。所以我**訓練**只用 CFEVER 的 train，不加任何外部資料集，也不自己生成合成資料。
2. **預訓練模型**：助教公告第 1 條本來就是「只能使用參數量 1B 以下的 **pre-trained model**」，代表預訓練模型是允許的（bert-base-chinese 本身也是用外部語料預訓練的）。三個模型都 < 1B，程式裡用 `count_parameters` 檢查。

Gemini 建議的「用 OCNLI / CMNLI 外部 NLI 資料集再訓練」就屬於第 1 點，所以不做。

---

### Q6. 拿 test 的 claim 去做標題比對、建候選池，算不算偷看答案？

不算。系統本來就一定要讀 test 的 claim 才能預測。程式**只讀 test 的 `id` 和 `claim`**（`read_jsonl(TEST_PATH)[['id', 'claim']]`），完全沒有 test 的標籤和證據。用 claim 去找候選頁，就是正常的「推論」。

---

## 📅 2026-11-11：資料與 Python 語法

### Q7. `evidence` 三層中括號怎麼讀？`gold_groups` 為什麼要寫這麼多檢查？

```text
evidence = [                                          ← 第 1 層：好幾組都可以（找到任一組就算對）
    [ [101, 1, '周杰倫', 0] ],                         ← 第 2 層：一組（這組 1 句）
    [ [101, 2, '周杰倫', 3], [101, 3, '葉惠美', 1] ],  ← 這組要 2 句一起找到
]                       ↑ 第 3 層：[標註 id, 證據 id, 頁名, 句號]
```

`gold_groups` 把它變成 `[{('周杰倫', 0)}, {('周杰倫', 3), ('葉惠美', 1)}]`，之後用 `<=`（issubset）判斷。

要檢查是因為：
- NEI 的證據長 `[[[101, None, None, None]]]`，頁名是 `None`，要跳過。
- **空集合陷阱**：`set() <= 任何集合` 永遠是 `True`。如果讓空組混進來，沒找到任何證據也會被算成命中。

練功本第 1 節可以直接跑看看。

---

### Q8. 為什麼 `wiki_pages.jsonl` 要讀兩次？讀一次全部存起來不行嗎？

維基有一百多萬頁、約 1 GB。全部的 `text` 和 `lines` 存成 Python 字串，記憶體會吃好幾 GB，Colab 免費版（約 12 GB RAM）有機會爆掉。

所以分兩次：
1. **第 1 次只留頁名** → 建立「標題 → 頁面」索引，做標題比對，算出「候選池」需要哪些頁。
2. **第 2 次只存候選池的頁面內容**（約十萬頁，而不是一百多萬頁）。

多讀一次檔大約多花 1～2 分鐘，換來記憶體少很多，很划算。

---

### Q9. `defaultdict(list)` 和 `dict.get(key, [])` 差在哪？為什麼建完索引又 `dict(title_index)`？

- `defaultdict(list)`：`d[key]` 查不到時，**會自動建立** `d[key] = []`。適合「建表」，可以直接 `d[key].append(x)`，不用先判斷 key 在不在。
- `d.get(key, [])`：查不到時**只回傳** `[]`，**不會**把 key 塞進字典。適合「查表」。

陷阱：如果建完索引後還用 `title_index[子字串]` 去查，`defaultdict` 會幫每個查不到的子字串都建一個空 list，字典會越查越大。所以建完就 `dict(title_index)` 轉回一般 dict，查詢一律用 `.get(..., [])`。

C# 的對照：`defaultdict` 像是自己包一個 `GetOrAdd`，`.get` 像 `TryGetValue` 失敗時回傳預設值。

---

### Q10. `re.sub(r'\s*[（(][^（）()]*[）)]\s*$', '', title)` 怎麼讀？

拆開來看：

| 片段 | 意思 |
| :--- | :--- |
| `\s*` | 0 個以上的空白 |
| `[（(]` | 一個左括號，全形或半形都可以 |
| `[^（）()]*` | 0 個以上「不是括號」的字（括號裡的內容） |
| `[）)]` | 一個右括號 |
| `\s*$` | 可能還有空白，然後就是字串結尾 |

所以「水星 (行星)」→「水星」、「水星（神話）」→「水星」，但「台北101」不受影響。`r'...'` 是 raw string，讓 `\s` 不會被 Python 當成跳脫字元。

---

### Q11. 標題比對是雙層迴圈，會不會很慢？

claim 平均幾十個字。起點有 n 種，長度最多 29 種（2～30），一題最多查約 30n 次，3 萬多題加起來約幾千萬次 dict 查詢。dict 查詢平均 O(1)，Python 大概十幾秒就跑完。

反過來，如果是「每個標題去 claim 裡找」（`for title in 一百多萬個標題: if title in claim`），就是 3 萬 × 100 萬 = 300 億次，完全跑不完。**「拿 claim 的子字串去查表」而不是「拿標題去 claim 裡找」**，是這段的關鍵。

---

### Q12. `sorted(found, key=lambda p: (-found[p], '(' in display_title(p), p))` 在排什麼？

`key` 回傳一個 tuple，Python 會依序比較 tuple 的每一項：

1. `-found[p]`：命中長度取負號 → 長的變成比較小的數，排在前面（等於由長到短）。
2. `'(' in ...`：`False` 比 `True` 小 → 沒有消歧義括號的頁排前面（「水星」比「水星 (神話)」優先）。
3. `p`：前兩項都一樣時，照頁名排，讓結果固定、可重現。

用負號而不用 `reverse=True`，是因為只想讓第 1 項反過來，第 2、3 項維持由小到大。

---

### Q13. `parts[0].isdecimal()` 為什麼不用 `isdigit()`？句號為什麼不能重新編號？

`isdigit()` 對「²」這種上標數字也會回傳 `True`，但 `int('²')` 會出錯；`isdecimal()` 只接受真正能轉成整數的數字，比較安全。

句號不能重新編號是因為**評分是比對原始句號**。例如第 1 句是空的被跳過，第 2 句還是第 2 句；如果自己用 `enumerate` 重新從 0 編號，所有證據都會錯開一格，Evidence Recall 直接變 0。

---

### Q14. `any(group <= found for group in groups)` 是什麼寫法？

- `group <= found`：集合的「子集合」判斷，等於 `group.issubset(found)`。
- `(... for group in groups)`：generator expression，一個一個產生 True / False。
- `any(...)`：只要有一個是 True 就回傳 True（而且一找到就停，不會全部算完）。

合起來就是「**任一組**標準證據**全部**被找到」，剛好就是 Recall 和 FEVER score 的命中定義。

---

## 📅 2026-11-11：向量檢索（bge-m3）

### Q15. 為什麼 bge-m3 取 [CLS] 向量，不像 sample code 做 mean pooling？什麼是 anisotropy？

**要照模型訓練時的用法。** bge-m3 用對比學習訓練時，就是拿第 0 個位置（[CLS]，XLM-R 叫 `<s>`）的向量去拉近「相關的句子」、推遠「不相關的句子」，所以它的 [CLS] 向量最適合拿來比相似度。換成 mean pooling 不是不能用，但不是它被訓練的方式。

sample code 的 bert-base-chinese 只做過 MLM 預訓練，**沒有**為「找相似文字」訓練過。研究發現這種原生 BERT 的向量會擠在一個很窄的錐形區域裡（**anisotropy，各向異性**）：隨便兩句話的 cosine 都很高，分不出好壞。這是 sample code 檢索分數低的主因之一。

`encode_texts` 兩種寫法都保留（`pooling='cls'` / `'mean'`），練功本第 3 節可以手打 mean pooling。

---

### Q16. 為什麼 L2 normalize 之後，內積就等於 cosine similarity？

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

`F.normalize(x, p=2, dim=1)` 把每個向量除以自己的長度，長度變成 1。分母變成 1 × 1，所以 cosine = 內積。

好處是可以把「所有 claim × 所有頁面」的 cosine 用**一個矩陣乘法** `claim_emb @ page_emb.T` 一次算完，GPU 最擅長這種運算。

---

### Q17. `encode_texts` 為什麼先把文字依長度排序？`vectors[index] = ...` 怎麼放回原位？

同一批（batch）的文字要 padding 到這批最長的長度。如果長短隨機混在一起，一批裡只要有一句很長，其他短句都要補一堆 `[PAD]`，GPU 在白算。先排序，同一批長度差不多，padding 很少，速度可以快好幾倍。

`order` 記錄「排序後第 k 個是原本的第幾筆」。每批算完，用 `vectors[index] = pooled` 把結果寫回**原本的位置**（`index` 是原本的編號 list，PyTorch 支援用 list 一次寫入多列）。所以函式回傳的順序還是和輸入一樣，呼叫的人不用管排序這件事。

---

### Q18. fp16 是什麼？為什麼向量存在 CPU 還用 fp16？

一般浮點數是 32 位元（fp32），fp16 是 16 位元：**記憶體少一半、GPU 算得更快**（T4 有專門的 fp16 運算單元），代價是精度比較低。檢索和推論對這點精度不敏感。

句子向量可能有幾十萬句 × 1024 維。fp32 要好幾 GB，fp16 只要一半，Colab 比較不容易爆 RAM。真的要算分數時再 `.float()` 轉回 fp32。

訓練分類器用的是**混合精度**（Q31），跟這裡只存 fp16 不太一樣。

---

### Q19. `page_candidates` 為什麼一次只拿 256 個 claim 去乘？

相似度矩陣的大小是「claim 數 × 候選池頁數」。3 萬個 claim × 10 萬頁 = 30 億個數字，fp16 也要 6 GB，GPU 會爆。每次 256 個 claim，矩陣只有 256 × 10 萬，算完馬上取 `topk`、丟掉矩陣，再算下一批。

---

### Q20. title bonus 是什麼？為什麼是「加分」，不是「標題命中的頁一律排前面」？

最終分數 = cosine + bonus × [這頁是不是被標題比對命中]（`True` 當 1、`False` 當 0）。

- bonus = 0：完全不管標題，只看向量。
- bonus 很大（例如 1.0）：標題命中的頁一定排在最前面。
- 中間值：標題命中是「加分」，但如果向量覺得它很不像，還是可能被別的頁超過。

為什麼不直接一律優先？因為 claim 裡常常會出現國名、年份、「電影」「歌手」這種**很通用的短標題**，它們也會被比對到。給太多分，這些通用頁就會把真正的主角頁擠出前 5 名。所以我讓 dev 決定 bonus 要多少（`TITLE_BONUS_GRID`），結果和整張表會自動寫進報告第 4 題。

Gemini 說的 `0.4 × BM25 + 0.6 × Dense` 是同一種想法（混合兩種分數），但 BM25 分數可能是 0～30、cosine 是 -1～1，兩個尺度差太多，直接加權會失真。我的「是否命中」只有 0 / 1，就沒有這個問題。

---

### Q21. 在 dev 上挑 bonus 和 NEI bias，再用同一份 dev 報告分數，算作弊嗎？

不算作弊（沒有用到 test 的答案），但 dev 分數會**稍微偏樂觀**，因為參數是照 dev 挑的，等於有一點點「對 dev 過擬合」。

這裡只挑兩個參數、候選值也不多，影響很小。報告第 4 題 (5) 已經老實寫出這點，並提出比較嚴謹的做法：把 dev 切兩半，一半挑參數、一半報告。

---

### Q22. `topk` 回傳的 `values` 和 `indices` 是什麼？

`similarities.topk(k=30, dim=1)` 沿著每一列（每個 claim）找最大的 30 個：
- `values`：這 30 個分數。
- `indices`：它們在那一列的**欄位置**，也就是候選池的第幾頁。要用 `pool_pages[index]` 換回頁名。

cosine 可能是負數，`topk` 只看排名，負數一樣會被選進來（候選不夠多的時候）。

---

## 📅 2026-11-11：證據檢索（reranker）

### Q23. 為什麼要在句子前面加頁名？

維基的句子常常用代名詞指主角，例如「葉惠美」頁裡寫「**該專輯**於 2003 年發行」。如果只看句子，模型根本不知道「該專輯」是哪張。加成「葉惠美：該專輯於 2003 年發行」，claim 裡的「葉惠美」就對得上了。分類器的輸入也一樣加了 `【頁名】`。

---

### Q24. 什麼是 local index、什麼是全域 index？為什麼這裡最容易寫錯？

所有候選句放在一個大 list `sentence_keys`。每個 claim 只看自己那 5 頁的句子，先收集它們在大 list 的位置 `rows`（例如 `[3, 4, 0, 1]`）。

對 `sentence_embeddings[rows]` 算分數、`topk` 之後，拿到的是**在 `rows` 裡的位置**（例如 `[1, 2]`，local）。要用 `rows[1]`、`rows[2]` 換回 `4`、`0`（全域），才能查 `sentence_keys`。直接拿 `[1, 2]` 去查，就會拿到別頁的句子，而且**不會報錯**，只會默默算錯分數。練功本第 5 節有對照示範。

---

### Q25. bi-encoder 和 cross-encoder 差在哪？為什麼不全部都用 cross-encoder？

| | bi-encoder（bge-m3） | cross-encoder（bge-reranker） |
| :--- | :--- | :--- |
| 怎麼算 | claim、句子**各自**變向量再比 | `claim + 句子`**一起**丟進模型 |
| 比喻 | HR 只比對兩份履歷 | 兩個人面對面聊，HR 在旁邊聽 |
| 成本 | 句子向量算一次，可以和所有 claim 比 | 每一對都要重跑一次模型 |

一題 5 頁可能有上百句，3 萬題全部用 cross-encoder 會跑好幾個小時。所以先用 bi-encoder 粗篩 20 句，再用 cross-encoder 精排，這是搜尋引擎的標準做法（retrieve → rerank）。

---

### Q26. 為什麼 reranker 只用在 dev / test，不用在 train？

train 的證據只拿來組分類器的訓練輸入：S / R 題反正會補進標準證據，NEI 題只需要「相關但不夠判斷」的句子。用向量初篩的前 5 句就夠了。省下來的是 2 萬多題 × 20 句的 cross-encoder 計算，大約 20～30 分鐘 GPU。

代價是 train 和 test 的「雜訊句」來源不完全一樣（初篩 vs 重排），但對分類器影響很小。

---

### Q27. 為什麼證據固定給滿 5 句？

FEVER score **只看前 5 句**，而且多給的句子不會扣分（不像 precision 會被拉低）。所以就算只有 1 句真的有用，給滿 5 句也能讓「標準證據剛好排第 4、5 名」的題目拿到分。

---

### Q28. 報告裡「文件已找對的題目中，證據也找齊的比例」在看什麼？

Evidence Recall 是「先找頁、再找句」的**整體**結果。如果它很低，可能是頁面沒找到（第 1 階段的錯），也可能是頁面對了但句子挑錯（第 2 階段的錯）。

只看「頁面已經找對」的題目再算一次，就能分開這兩種原因：
- 這個比例很高 → 證據 Recall 低主要是第 1 階段害的。
- 這個比例也低 → 挑句子本身還有問題。

報告第 4 題的「三階段漏斗」表也是同樣的想法，把每一步少掉幾題列出來。

---

## 📅 2026-11-11：分類器

### Q29. 為什麼換成 `hfl/chinese-roberta-wwm-ext`？要不要用 large？

它和 bert-base-chinese 一樣大（約 102M），差別在預訓練方式：
- **全詞遮罩（Whole Word Masking）**：中文遮的是整個詞（「周杰倫」三個字一起遮），不是單一個字，模型比較學得到詞的意思。
- 用更多中文語料、訓練更久（RoBERTa 的做法）。

在中文推論任務上通常比 bert-base-chinese 好一點，而且 tokenizer 和 bert-base-chinese 相同，程式完全不用改。

large 版（`hfl/chinese-roberta-wwm-ext-large`，約 326M）會再好一些，但訓練時間大約 3 倍，而且 large 模型有時候會訓練不穩。這次先用 base；時間夠的話，只要改 `CLASSIFIER_NAME`，必要時把 `BATCH_SIZE` 調成 8。

---

### Q30. 訓練時把標準證據塞進去，算不算作弊？

不算。**train 的標準證據本來就是訓練資料的一部分**，就像 HW2 用 train 的答案算 loss 一樣。不能碰的是 dev / test 的答案：dev 和 test 的輸入**全部**是檢索出來的證據，沒有塞任何標準答案。

為什麼要塞？FEVER score 規定 SUPPORTS / REFUTES 的證據沒找到就是 0 分，所以分類器真正需要學的是：
1. 證據對的時候，判斷 S 還是 R；
2. 證據不夠的時候，選 NEI。

sample code 直接拿檢索結果訓練，檢索錯時等於叫模型看著無關的句子學「這是 SUPPORTS」，學到的是雜訊。

最後的 `rng.shuffle(keys)` 是為了不讓模型偷懶記住「標準證據總是在第 1 句」。

---

### Q31. `torch.autocast` 和 `GradScaler` 在做什麼？

這是**混合精度訓練**：
- `torch.autocast(..., dtype=torch.float16)`：前向傳播時，矩陣乘法這類運算自動用 fp16（快），容易出問題的運算（例如 softmax、loss）自動維持 fp32。權重本身還是 fp32。
- `GradScaler`：fp16 能表示的最小數字有限，很小的梯度會直接變成 0（underflow）。所以先把 loss 乘上一個大數字再 backward（`scaler.scale(loss).backward()`），梯度跟著被放大；更新前再除回來（`scaler.unscale_`），然後才做梯度裁剪。如果發現梯度變成 inf，`scaler.step` 會跳過這一步，並自動調小放大倍率。

在 CPU 上 `USE_FP16 = False`，兩個都會自動關掉，程式不用改。

---

### Q32. warmup 是什麼？為什麼前 10% 要慢慢升學習率？

`get_linear_schedule_with_warmup` 讓學習率：前 10% 的步數從 0 線性升到 2e-5，之後線性降到 0。

分類層是新接上去、隨機初始化的，剛開始的梯度很大又很亂。如果一開始就用全速的學習率，可能把預訓練好的權重弄壞（catastrophic forgetting）。先小步走，等分類層穩定一點再加速。練功本第 8 節有印出每一步的學習率。

---

### Q33. `truncation='only_second'` 和 `DataCollatorWithPadding` 是什麼？

- `truncation='only_second'`：輸入是句對（claim, 證據）。太長時**只截第二段（證據）**，claim 一個字都不截。claim 是要判斷的主體，截掉就沒意義了。
- `DataCollatorWithPadding`：sample code 是每一筆都 padding 到固定的 192；這裡改成 **dynamic padding**，每一批只補到「這批最長」。大部分批次比上限短很多，可以省不少時間。

---

### Q34. 每個 epoch 結束都在 dev 上評估，會不會影響訓練？

不會。`predict_logits` 有 `@torch.no_grad()`，也有 `classifier.eval()`，不算梯度、不更新參數。下一個 epoch 開頭會再呼叫 `classifier.train()` 切回訓練模式（開 dropout）。

這張 epoch 表會寫進報告第 3 題，可以看出第 2 個 epoch 有沒有比第 1 個好。

---

### Q35. NEI bias 為什麼可能讓 FEVER score 上升、Accuracy 下降？

在 NEI 的 logit 加一個正數 → 模型更常猜 NEI。

看一題「正確是 SUPPORTS、但證據沒找到」的題目：
- 猜 SUPPORTS：Accuracy 算對，**FEVER 還是 0 分**（證據沒找到）。
- 改猜 NEI：Accuracy 算錯，FEVER 一樣 0 分。

對 FEVER 來說兩個一樣，但「多猜 NEI」可能讓真正的 NEI 題多拿分。所以 bias 往上調，FEVER 可能上升，Accuracy 卻下降。競賽看 FEVER，所以 test 用 FEVER 最高的 bias。如果 dev 結果是 bias = 0 最好，報告也會自動寫「沒有調整」。

---

### Q36. Accuracy、本地 FEVER score、Kaggle 分數三個是什麼關係？

| | 在哪算 | 看什麼 |
| :--- | :--- | :--- |
| Accuracy（報告 Q3） | dev | 只看標籤 |
| 本地 FEVER score | dev | 標籤 + 前 5 句證據（NEI 只看標籤） |
| Kaggle public / private | test（助教那邊算） | 競賽的 FEVER score |

本地 FEVER 是用來**估計** Kaggle 分數，兩者會接近但不會一樣（test 和 dev 的題目不同）。報告的 public / private 分數一定要填 Kaggle 上的真實分數，不能拿 dev 的數字代替。

---

## 📅 2026-11-11：執行與繳交

### Q37. 在 Kaggle 上怎麼跑？要注意什麼？

詳細步驟在 [index.md](index.md)。重點：
- 右側 **Add Input** 加入競賽資料；**Settings → Accelerator** 選 GPU（T4 或 P100）；**Internet 要開**（要下載 3 個模型，約 5 GB）。
- 每週 GPU 上限 30 小時，完整跑一次大約 1 小時，夠用。
- 跑完 `/kaggle/working/submission.csv` 會出現在右側 Output，從那裡提交。
- 如果競賽要求「離線提交 notebook」，要先把 3 個模型加成 Kaggle Dataset 並加入 Input，`resolve_model` 會自動找到它們（資料夾名稱要包含 `bge-m3`、`bge-reranker-v2-m3`、`chinese-roberta-wwm-ext`）。

---

### Q38. 在 Colab 跑到一半斷線怎麼辦？

第 3 節最後會把檢索結果存成 `retrieval_cache.json`（存在 Drive 的 `nlp-hw3/output/`）。如果是在訓練分類器時斷線：

1. 重新連線，執行第 1 節（到「讀取資料」那格為止），再執行第 2 節第一格（`gold_groups`、`recall_at` 這些工具函式）和「用 bge-m3 把文字變成向量」那格（裡面有 `resolve_model`）。v2 已經把 `resolve_model` 搬進第 2 節第一格，所以 v2 不用跑 bge-m3 那格。
2. 找到「Colab 斷線救援」那一格，把 `LOAD_RETRIEVAL_CACHE = False` 改成 `True`。
3. 從那一格往下執行，就不用重跑 30 分鐘的檢索。

如果是檢索跑到一半就斷，那只能從頭跑。

---

### Q39. 出現 `CUDA out of memory` 要改哪個參數？

看是哪一格爆的：

| 爆在哪 | 改這個 |
| :--- | :--- |
| 編碼頁面 / 句子 | `EMBED_BATCH_SIZE` 64 → 32 |
| reranker | `RERANK_BATCH_SIZE` 64 → 32 |
| 訓練分類器 | `BATCH_SIZE` 16 → 8 |

改完要從那一節重新執行。只要 batch size 改了、模型和資料沒改，結果應該只有很小的差異。

---

### Q40. 我想先快速試跑整條流程，確認不會出錯，要怎麼設？

在參數格改：

```python
TRAIN_SAMPLES = 2000
DEV_SAMPLES = 300
TEST_SAMPLES = 300   # 不是 None 時不會寫出 submission.csv，避免把試跑結果交出去
EPOCHS = 1
```

試跑的數字**不要拿去寫報告**。確認沒問題後全部改回 `None` / `2`，再完整跑一次。維基一樣會整份讀兩次，這部分不會因為試跑而變快。

---

### Q41. 報告要怎麼填？

1. 完整跑完後，第 7 節會印出**整份 Report**（已經填好這次的數字和 5 張表），同時存成 `output/report_filled.md`。
2. 把它整段複製，貼到最下面的 Report markdown 格子（取代原本有【待填】的版本）。題目文字和格式都沒動，只填了 Answer。
3. 提交 `submission.csv` 到 Kaggle，把 public score 填到 `Your public leaderboard score:` 後面。
4. 11/23 12:00 private 公布後，再填 private score。
5. 自己讀一遍。第 5 題的 AI 使用聲明如果和實際情況不同，要自己改。

---

### Q42. 最後要交什麼？檔名是什麼？

- 把 `st_minecode.ipynb`（**保留執行輸出**、Report 已填好）改名成 **`NLP_HW3_NTOU_11457088.ipynb`**，再壓成 **`NLP_HW3_NTOU_11457088.zip`**，上傳 NTU COOL（11/25 23:59 前）。學校和學號照 HW2 的檔名，交之前再核對一次。
- 檔名錯：`.ipynb` 和 `.zip` 各扣 5 分。
- `Your Kaggle username:` 要和 **leaderboard 上顯示的名字**一樣（助教連說三次）。目前填的是 `zhi816`，要到 Kaggle 排行榜確認。
- Kaggle 截止前（11/23 12:00）要**自己選最多 2 個 submission**，記得選這版的。
- 練功本、`st_question.md`、`index.md` 是給自己讀的，不用交。

---

### Q43. 如果我改了參數（例如 `DOCUMENT_TOP_K = 10`），報告會跟著變嗎？

第 7 節印出的報告，數字和大部分設定（top-k、候選數、learning rate、epoch…）都是從變數讀的，會跟著變。少數寫死在文字裡的描述（例如「子字串長度 2～30」、模型名稱）如果你改了 `MIN_TITLE_LEN` / `MAX_TITLE_LEN` 或換了模型，要記得自己改。`USE_RERANKER = False` 時，Q2 的 (2) 那一點也要自己刪掉。

---

## 📅 2026-11-11：v1 拿到 0.776（第 64 名）之後 — v2

### Q44. v1 是 0.776、排第 64，分數卡在哪？v2 為什麼選這幾個方向？

FEVER score 是把每一題的得分加起來：

```text
SUPPORTS / REFUTES 題：① 前 5 頁有找到 → ② 前 5 句有找齊 → ③ 標籤也對      → 1 分
NOT ENOUGH INFO 題：  標籤猜 NEI                                          → 1 分
```

所以失分只會來自 4 個地方：文件沒找到、句子沒找齊、證據對了但標籤錯、NEI 題被猜成 S/R。v1 第 5 節的**三階段漏斗表**會告訴你每一步掉了幾題。（老實說，我寫 v2 的時候還看不到 v1 實際的 dev 數字，所以 v2 是依照 CFEVER 常見的失分點來設計的。）

- **文件檢索**：v1 用完整維基的標題比對，CFEVER 的 claim 大多會直接寫出主角名稱，所以這一段通常已經很高，v2 **不動它**，避免改壞。
- **證據檢索**：v1 的 reranker 是現成的，沒看過 CFEVER → v2 用 train 微調它（Q46）。
- **分類**：v1 是 base 模型，而且輸入可能把證據截掉 → v2 換 large、放寬長度（Q50、Q51）。
- **NEI**：v1 只有 NEI bias → v2 多一個 reranker 門檻（Q54）。

v2 第 7 節會自動做 v1 / v2 對照表（只要讀得到 v1 的 `metrics.json`），哪一段進步最多一看就知道，也可以直接寫進報告第 4 題。

---

### Q45. v2 還符合作業規範嗎？

逐條對照 [作業規範.md](作業規範.md)：

| 規定 | v2 的做法 |
| :--- | :--- |
| 每個模型 < 1B | bge-m3 568M、bge-reranker-v2-m3 568M、chinese-roberta-wwm-ext-large 326M；`count_parameters` 會 assert |
| 禁止外部推理 API | 全部在 Kaggle / Colab 的 GPU 上跑，只有從 Hugging Face **下載權重**（規範寫明不算 API） |
| 只用 CFEVER dataset | reranker 和分類器都只用 CFEVER **train** 訓練；dev 只用來挑參數、選 epoch、選 reranker；test 只讀 `id`、`claim` |
| 預訓練 checkpoint | 11/06 答疑允許、11/11 沒有撤回（規範第三節）；三個都是公開的預訓練模型，來源連結寫在程式最上方 |
| Report 格式不能改 | 題目文字一字未動，只在 `Answer:` 後面作答 |
| AI 使用要註明 | 程式最上方註解 + Report 第 5 題都有寫 v2 的部分 |

我刻意**沒有用**「已經在其他 NLI 資料集上微調過的分類模型」（例如 Erlangshen-NLI 系列）。它們是用別的標註資料集訓練的，和 11/11「只能用 CFEVER」的答覆比較衝突，所以分類器選的是只做過語言模型預訓練的 `chinese-roberta-wwm-ext-large`。

---

### Q46. 微調 reranker 到底在做什麼？為什麼是 1 個正例 + 7 個負例？

現成的 bge-reranker 學的是一般的「這句和問題相不相關」。CFEVER 要的是「**能拿來判斷真假的那一句**」，而且標註者挑哪一句有自己的標準。用 train 的標準證據教它，它就會往 CFEVER 的標準靠。

一題組成一份 8 選 1 的「考卷」：

```text
claim：周杰倫在 2003 年發行葉惠美
  第 0 句（正例）：葉惠美：葉惠美是周杰倫在 2003 年發行的專輯     ← 標準證據
  第 1～7 句（負例）：周杰倫：周杰倫是臺灣歌手 …                 ← 很像但不是
```

reranker 對 8 句各打一個分數，把 8 個分數當 logit 做 cross entropy，答案永遠是第 0 個。這叫 **listwise loss**，比「每句單獨判斷是不是證據」更直接：我們在乎的只是**排名**（標準證據要排第 1），不在乎分數的絕對值。7 是一個常見的設定，負例越多越難、但記憶體也越多。

---

### Q47. 什麼是 hard negative？會不會把「其實也是證據、只是沒被標」的句子當成負例？

hard negative = **看起來很像答案、其實不是**的句子。v2 從 bge-m3 初篩的前 14 名（已排除所有標準證據句）裡隨機抽 7 句。如果隨便抽「富士山是日本最高的山」當負例，模型一看就知道不相關，學不到東西；同一頁、同一主題的句子才會逼它學會細微的差別。

會不會抽到「其實也能當證據」的句子？會，這叫 **false negative**，是所有檢索訓練都會遇到的雜訊。影響通常不大：一來它只佔一小部分；二來 listwise loss 只要求標準證據**比較高**，不會把那句的分數壓到很低。也因為有這種風險，v2 會在 dev 上比較微調前後，變差就不用（Q49）。

---

### Q48. 為什麼要凍結詞向量層？這樣 reranker 會不會變笨？

XLM-R 的詞表有 25 萬個詞，詞向量層就佔了約 2.5 億參數（整個模型的 4 成多）。訓練時每個參數要存權重、梯度和 Adam 的 2 個狀態，大約 16 bytes：

- 不凍結：568M × 16 ≈ 8.5 GB，加上 activation，T4 的 16 GB 很吃緊。
- 凍結後：可訓練的約 3.1 億 × 16 ≈ 4.6 GB，放得下。

不會變笨：詞向量代表「每個字詞是什麼意思」，這在預訓練時已經學好了；我們要教的是「怎麼比對 claim 和句子」，那是後面 24 層 transformer 的工作。練功本 §5 可以自己算一次參數量。

---

### Q49. 微調完反而變差怎麼辦？

程式會自動處理：微調前先用**現成的** reranker 在 dev 上算 Evidence Recall@5（`zs_recall`），微調後再算一次（`ft_recall`）：

- `ft_recall >= zs_recall` → 用微調後的。
- 否則 → `load_reranker()` 重新載入現成的，test 用現成的。

dev 沒有拿來訓練 reranker，所以這個比較是公平的。最差的情況就是和 v1 一樣用現成的 reranker（多了 30 句初篩），不會更糟。報告第 2 題會自動寫最後用的是哪一個。

---

### Q50. 為什麼分類器要換 large？會不會太慢、太佔記憶體？

`chinese-roberta-wwm-ext-large` 和 v1 用的 base 是同一系列，層數 12 → 24、hidden 768 → 1024、參數 102M → 326M，在中文推論任務上通常比 base 高 1～3 個百分點，而且仍遠小於 1B。

代價是訓練大約慢 3 倍，再加上長度 256 → 512，所以 v2 的分類器大約要 30～40 分鐘（T4）。記憶體方面：

- 一次放 8 筆，用 gradient accumulation 累積 2 批才更新（Q52），等效 batch size 還是 16。
- 還是 `CUDA out of memory` 的話：`BATCH_SIZE = 4`、`GRAD_ACCUM_STEPS = 4`，或打開 `GRADIENT_CHECKPOINTING = True`。
- large 的學習率用 `1e-5`（base 是 `2e-5`）。large 模型偶爾會「學不起來」（dev Accuracy 卡在 40% 左右、一直猜同一類），程式會印警告，這時把學習率降到 `5e-6` 或換 `SEED` 重跑。

---

### Q51. v1 的「證據被截掉」是怎麼回事？把標準證據放最前面，模型會不會偷懶只看第 1 句？

v1 的輸入是 `claim + 5 句證據`，上限 256 token。中文 BERT 大約一個字一個 token，5 句維基句子加上頁名很容易超過 256，**超過的部分直接丟掉**。v1 訓練時又把 5 句打亂，標準證據有機會排到第 4、5 句，剛好被丟掉；這時模型看不到證據，卻被要求回答 SUPPORTS / REFUTES，學到的就是雜訊。

v2 第 4 節會用 tokenizer **實際量**「v1 設定下有多少比例的標準證據被截掉」，這個數字可以直接寫進報告第 4 題。

v2 的三個修正：上限 512、每句最多 120 字、標準證據放前面。會不會讓模型只看第 1 句？

- 測試時 reranker 也把最像證據的句子排第 1，所以「訓練時看到的樣子」和「測試時看到的樣子」一致，這正是我們要的。
- 需要 2 句的題目，2 句標準證據都會放在最前面。
- NEI 題的第 1 句也是很相關的句子（reranker 挑的），模型沒辦法靠「第 1 句長得像不像」來作弊，還是要真的讀懂它能不能判斷真假。

---

### Q52. gradient accumulation 和直接 batch 16 一樣嗎？

梯度**完全一樣**（練功本 §7 會用程式驗證）。PyTorch 的梯度是累加的：兩批各自 `backward()`，`.grad` 就是兩批的梯度相加。每批的 loss 先除以 2，相加後就等於 16 筆的平均。差別只在：

- 比較慢一點（同樣的資料分兩次跑）。
- 學習率排程要用「更新次數」來算：`updates_per_epoch = ceil(批數 / 2)`，不是批數。
- BatchNorm 類的層會不一樣，但 BERT 用的是 LayerNorm，不受影響。

---

### Q53. 保留 dev 最好的 epoch，為什麼存權重要 `.clone()`？這樣用 dev 算不算作弊？

`model.state_dict()` 回傳的 tensor 和模型**共用記憶體**。如果只寫 `best = model.state_dict()`，模型繼續訓練，`best` 也會跟著變，最後存到的其實是最後一個 epoch。所以要 `.detach().cpu().clone()` 複製一份到 RAM（練功本 §8 有示範）。最後一個 epoch 就是現在的權重，不用另外存，省 1.3 GB。

用 dev 選 epoch 和用 dev 挑 bonus、NEI bias 一樣，是正常的 model selection（見 Q21）。dev 從頭到尾沒有拿來算梯度。只是報告裡的 dev 分數會略為樂觀，報告第 4 題 (7) 有寫這個限制。

---

### Q54. 「reranker 門檻」是什麼？為什麼用分位數，不直接寫一個分數？

reranker 對每一題的候選句打分數，**第 1 名的分數**代表「最像證據的那一句有多像」。如果連第 1 名都很低，代表證據多半沒找到：

- 如果這題其實是 S/R：證據沒找到，猜對標籤也是 0 分，改猜 NEI 沒有損失。
- 如果這題其實是 NEI：改猜 NEI 就多拿 1 分。

所以「第 1 名分數低於門檻就判 NEI」有機會提升 FEVER score。它和 NEI bias 的差別是：NEI bias 看的是**分類器**的想法，門檻看的是 **reranker** 的想法，兩個訊號互補。

為什麼用分位數？reranker 的分數是 logit，範圍會因為模型（現成的或微調後的）而不同，直接寫「分數 < 0」不一定合理。用「dev 上分數最低的 10%」這種說法，不管分數範圍是多少都適用。挑好之後，換算成實際的分數值（`best_threshold`）套用到 test。門檻和 bias 的所有組合都在 dev 上試過（grid search），如果加門檻沒有比較好，程式會選「不使用」。

---

### Q55. 為什麼 train 的 NEI 題也要用 reranker 挑證據？

分類器要學的是「這幾句**夠不夠**判斷 claim」。測試時，NEI 題的證據是 reranker 從 30 句裡挑出最像的 5 句：主題完全對、看起來很相關，但其實資訊不夠。v1 訓練時，NEI 題的證據是向量隨便取的前 5 句，比測試時「好認」很多。模型在簡單的題目上學會 NEI，遇到測試時那種很像的句子，就容易誤判成 SUPPORTS / REFUTES。

v2 讓 train 的 NEI 題也經過 reranker（從前 10 句挑 5 句，省時間），訓練和測試的難度就一致了。S/R 題不用：它們會補進標準證據。

---

### Q56. v2 要跑多久？為什麼建議用 Kaggle 的 Save & Run All？

T4 估計 1.5～2 小時（細項在 [index.md](index.md) 第八節）。Colab 免費版閒置或跑太久容易斷線，而且分頁要一直開著。Kaggle 的 **Save Version → Save & Run All (Commit)** 會在背景用一台新的機器從頭跑到尾：

- 關掉瀏覽器、電腦睡眠都不影響。
- 單次最多 12 小時，v2 綽綽有餘；每週 GPU 有 30 小時，可以跑好幾次。
- 跑完後到該版本的 **Output** 下載 `submission.csv` 和 `report_filled.md`，notebook 的執行輸出也會保留下來，交作業時直接下載那個版本的 `.ipynb`。

想先確認不會出錯，可以照 Q40 試跑（`TRAIN_SAMPLES = 2000`、`DEV_SAMPLES = 300`、`TEST_SAMPLES = 300`、`EPOCHS = 1`），v2 另外可以把 `RERANK_TRAIN_CLAIMS` 改成 500。

---

### Q57. 如果 v2 沒有比 v1 好怎麼辦？最後 2 個 submission 怎麼選？COOL 交哪一份？

- **先看 dev**：v2 第 7 節的 v1 / v2 對照表，比較兩版的 dev FEVER score。dev 有 3,000 題，比 public leaderboard 的部分測試資料穩定。
- **Kaggle 最終 2 個**：建議選「public 最高的那份」+「dev FEVER 最高的那份」；如果是同一份，第 2 個選另一版當保險。private 分數用的是另一部分測試資料，public 最高不一定 private 也最高。
- **COOL 只交 1 份 notebook**：交分數比較好的那一版，改名成 `NLP_HW3_NTOU_11457088.ipynb`。Report 裡的 public score 要填**這一版**的分數。如果交 v2，第 4 題已經有 v1 / v2 對照，內容比較完整。

---

### Q58. 還想再往上衝，有什麼方法？

依「預期效果 ÷ 花的 GPU 時間」排序，都要遵守作業規範：

1. **train + dev 合起來訓練最終分類器**：先用 v2 在 dev 上挑好參數（bias、門檻、epoch 數），再把 dev 也加進訓練資料重訓一次，test 用這個模型。資料多 12%，通常能再進步一點，但就沒有 dev 可以檢查了。
2. **多個 seed 的 ensemble**：同樣設定換 2～3 個 `SEED` 各訓練一次，把 logit 平均。每個模型仍然 < 1B（規範是逐一計算）。時間 × 2～3。
3. **換分類器**：`hfl/chinese-macbert-large`（326M）和 roberta-large 差不多大，可以都試，挑 dev 比較好的，或兩個一起 ensemble。
4. **讓證據來自更多頁**：`DOCUMENT_TOP_K = 8`，報告 Recall@8。文件 Recall 會上升，但候選句變多、比較雜，要靠 reranker 挑得夠準。
5. **多句證據**：需要 2 句一起才能判斷的題目，reranker 一句一句看很吃虧。可以讓 reranker 看「第 1 名 + 候選句」的組合，但程式會複雜很多。

先等 v2 的漏斗表出來，看哪一段掉最多題，再決定要做哪一個。
