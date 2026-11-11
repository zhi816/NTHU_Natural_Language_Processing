# Assignment 3 學習問答

這份是依你的 Assignment 2 問題習慣預先整理的：從「這個變數是什麼」一路問到「為什麼分數很低」。要找執行步驟看 [index.md](index.md)，要逐段對照程式看 [程式逐段解析.md](程式逐段解析.md)。下面的小例子都是示意，不是本次實測結果。

## 一、作業到底要做什麼？

### Q1. Claim、document、evidence、label 各是什麼？

- **Claim**：要查證的一句話，例如「某城市位於甲國」。
- **Document**：可能包含相關資訊的 Wikipedia 頁面，例如城市頁面。
- **Evidence**：文件裡能支持或反駁 claim 的句子，輸出時需要頁名和原始句號。
- **Label**：`supports`、`refutes`、`NOT ENOUGH INFO` 三類之一。

系統不是只問分類器猜答案，還要先找到合理的頁面與句子。

### Q2. 為什麼要分成三個階段？

如果每題都讓分類模型讀整個 Wikipedia，內容太多也超過模型輸入長度。先找相關頁面，再從頁面找句子，最後才讓分類器看 claim 和少量證據。前一階段漏掉重要資訊，後面通常很難補救。

### Q3. `supports`、`refutes`、`NOT ENOUGH INFO` 的差別？

`supports` 是證據支持 claim，`refutes` 是證據反駁 claim，`NOT ENOUGH INFO`（NEI）是現有資訊不足以判斷。NEI 不等於「claim 一定是錯的」；此 notebook 對預測為 NEI 的題目輸出空證據 `[]`。

### Q4. 這份 notebook 有搜尋全部 Wikipedia 嗎？

沒有。它用全部 CFEVER 訓練資料標註過的頁面建立候選庫，再在其中找頁面。這樣比較省資源，但對正確頁面不在候選庫中的題目，Recall 有先天上限。程式會印出 dev 候選頁面涵蓋率來觀察這點。

### Q5. 訓練只抽 3000 筆，為什麼候選頁面還是使用全部訓練資料？

程式把兩件事分開：`all_train_df` 的全部公開訓練標註負責決定候選頁面；從中抽出的 `train_df` 只決定分類器的訓練樣本。這是目前的設計，不是說分類器用全部訓練資料訓練。

### Q6. 用 test 的 claim 找頁面算不算偷看答案？

正常預測一定要讀測試輸入的 claim，否則系統無法回答；程式建立 `selected_pages` 時也會納入 test claim 檢索到的頁面，以便編碼這些頁面的句子。它只讀 test 的 `id` 和 `claim`，不使用 test 的標準標籤或標準證據。不要把其他來源的測試答案加入流程。

## 二、資料與 Python 語法

### Q7. JSONL 是什麼？跟一般 JSON 有何不同？

JSONL 通常一行是一個獨立 JSON 物件。例如第一行是第一筆 claim，第二行是第二筆。`pd.read_json(file, lines=True)` 因此把每一行轉成 DataFrame 的一列；`wiki_pages.jsonl` 比較大，程式用 `for line in file` 逐行讀取。

### Q8. `[[[16, 15, '金朝', 34]]]` 有三層中括號，怎麼讀？

外層是「所有可用證據組」，中層是一組需要一起成立的證據，內層是一條證據的欄位。最內層的第 3、4 個元素分別是頁名 `金朝` 和原始句號 `34`。一組有多條時必須整組找到；有多組時找到任一完整組即可。

### Q9. `set()` 和 `list` 差在哪？為何頁名要放 set？

`list` 是有順序的清單，允許重複；`set` 是不重複元素的集合，適合去重和判斷包含關係。許多訓練題可能引用相同頁面，`wanted_pages` 用 set 可避免重複載入。後續要穩定對應向量列號時，再用 `sorted(wiki_pages)` 產生排序後的頁名清單。

### Q10. `issubset` 是什麼？為何要小心空集合？

`needed.issubset(found)` 表示 needed 的每個元素都在 found 裡。例如 `{'A', 'B'}.issubset({'A', 'B', 'C'})` 是 `True`。但 `set().issubset({'A'})` 也是 `True`；所以 `gold_groups` 會排除空組，避免完全沒有證據卻被算成命中。

### Q11. `for index, row in enumerate(dev_df.itertuples())` 每次拿到什麼？

`itertuples()` 逐列提供 DataFrame 的資料，`enumerate()` 再加上從 0 開始的計數。`row.label`、`row.evidence` 是該列欄位；`index` 可用來取 `dev_documents[index]`。抽樣後先 `reset_index(drop=True)`，並在 dev DataLoader 不打亂順序，才能保持資料對齊。

### Q12. `defaultdict(list)` 和一般字典有什麼不同？

它是一種字典；第一次存取尚不存在的 key 時，會用 `list()` 建立空清單。例如 `page_to_sentence_indices[title].append(i)` 在新頁名第一次出現時，不需先寫一段判斷來建立 `[]`。清單內容是句子在全體 `sentence_info` 中的索引。

### Q13. `for title in pages for index in ...` 是什麼語法？

它是巢狀 list comprehension，等價概念如下：

```python
candidate_indices = []
for title in pages:
    for index in page_to_sentence_indices.get(title, []):
        candidate_indices.append(index)
```

`.get(title, [])` 在頁名不存在字典時回傳空清單，避免直接索引造成 `KeyError`。它不會憑空建立新的證據句。

### Q14. 為什麼把 `[頁名, 句號]` 轉成 tuple？

Python list 可修改，不能當 set 的元素；tuple 可以用在這裡。因此 `tuple(['金朝', 34])` 變成 `('金朝', 34)`，才能和標準證據的 tuple set 做集合比較。輸出 CSV 時仍用 list 形式，因為提交格式需要 JSON 陣列。

### Q15. `zip()` 是解壓縮嗎？

視上下文而定。`zipfile.ZipFile` 是處理 ZIP 檔；Python 的 `zip(a, b)` 則是把兩個可迭代物件逐項配對。例如 `zip(['甲', '乙'], [1, 2])` 依序產生 `('甲', 1)`、`('乙', 2)`，不會處理壓縮檔。

## 三、Tokenizer、BERT 和向量

### Q16. Tokenizer、token ID 和 embedding 是同一個東西嗎？

不是。Tokenizer 先把文字切成模型使用的 token，並轉成整數 ID；BERT 收到 ID 之後，會在模型內部轉成向量並經過多層運算。ID 只是詞彙表的編號，不是可以直接拿來代表語意相似度的大小。

### Q17. `padding` 和 `attention_mask` 為什麼一起出現？

同一批文字長度不同時，padding 會補齊到相同長度，才能堆成矩陣。`attention_mask` 標示哪些位置是真實輸入、哪些是補位；檢索程式的 mean pooling 用這個 mask 排除補位，不讓填充位置影響平均。

### Q18. `unsqueeze(-1)` 到底加在哪裡？

原本 mask 形狀是 `[batch, token數]`，`unsqueeze(-1)` 在最後增加一維，變成 `[batch, token數, 1]`。這樣就能和 `[batch, token數, 768]` 的 token 向量相乘：每個 token 的 0 或 1 會套用到它的 768 個數值。這叫 broadcasting。

### Q19. 為什麼把 token 向量平均？`[CLS]` 可以直接拿來用嗎？

因為要把不定長度的文字變成固定維度向量，這份程式選用 mean pooling；它是簡單基線，不保證比其他 pooling 更好。也可以設計用 `[CLS]` 向量或其他句向量模型，但那是不同方法，必須實際比較。此處有效特殊 token 也包含在平均裡。

### Q20. `F.normalize` 做了什麼？為何之後可以用 `@`？

它在指定維度做 L2 正規化，使非零向量長度變成 1。Cosine similarity 是內積除以兩個向量的長度；兩者都已是單位向量時，分母是 1，所以內積等於 cosine。`@` 是 Python 的矩陣乘法運算子；不代表每次使用 `@` 都自動是 cosine，前提是向量已適當正規化。

### Q21. 文件相似度矩陣的每一格代表什麼？

如果這批有 2 個 claim、候選有 3 頁，矩陣形狀就是 `[2, 3]`。第 0 列第 2 欄是第 0 個 claim 與第 2 頁的向量相似度。`topk(dim=1)` 沿著每一列找最相近的頁面，而不是把不同 claim 混在一起排名。

### Q22. 為什麼句子檢索還要有 `local_index` 和 `index`？

句子排序只對「當前 claim 的候選句」計分，因此 `topk` 回傳的 `local_index` 是這份小清單的位置；真正要查 `sentence_info` 時，先用 `index = candidate_indices[local_index]` 轉成全體句子清單的位置。如果直接把前者當成後者，會拿錯頁名或句號。

### Q23. 凍結檢索 BERT，後面又訓練 BERT，矛盾嗎？

不矛盾：程式前後載入兩個模型實例。檢索模型只產生向量，不更新權重；分類模型另載入同一預訓練 checkpoint 並訓練三分類任務。分類器更新後，前面已經算好的檢索向量不會跟著更新。

### Q24. 句子超過長度限制會怎樣？

檢索 tokenizer 最多保留 `RETRIEVAL_MAX_LENGTH = 256` token；分類時 claim 和 context 配對後最多 `MAX_LENGTH = 192` token。context 在 tokenization 前還先截成 1000 個字元。重要資訊如果落在截斷處後面，模型就看不到；字元數和 token 數不是同一個單位。

## 四、模型訓練和評估

### Q25. `Dataset` 跟 `DataLoader` 分別做什麼？

`ClaimDataset` 定義怎麼取得單筆 token tensor 和訓練標籤；`DataLoader` 從 Dataset 取出多筆組成 batch，必要時打亂順序。訓練用 `shuffle=True`，dev / test 用 `False`，以保持預測順序與原資料對齊。

### Q26. `model.train()` 是真的在訓練嗎？

它只是切換到訓練模式，讓 dropout 等層使用訓練行為；真正的更新發生在先 `loss.backward()` 計算梯度、再 `optimizer.step()` 更新權重。相對地，`model.eval()` 切換評估行為，`torch.no_grad()` 停止建立梯度圖；兩者作用不同，推論時常一起用。

### Q27. `zero_grad()`、`backward()`、`step()` 次序能調換嗎？

這裡的標準順序是先清掉上一批梯度 → 前向算 loss → `backward()` 算這一批梯度 → 必要時裁剪 → `step()` 更新。若忘了 `zero_grad()`，梯度預設可能累積；有些方法會刻意累積，但這份程式不是那樣設計。

### Q28. Loss 下降是不是就保證 Kaggle 分數上升？

不是。Loss 反映目前批次的訓練目標；Kaggle 的分數還會受到 dev/test 分布、過擬合、文件檢索和證據是否正確影響。要比較模型，應看真實 dev 指標，最終仍以 Kaggle 的競賽評分為準。

### Q29. Recall@5 到底怎麼算？

這份 notebook 只計可驗證的 SUPPORTS／REFUTES 題目。文件 Recall@5 檢查前 5 頁是否涵蓋任一完整證據組需要的所有頁面；證據 Recall@5 檢查前 5 句是否涵蓋任一完整證據組的所有頁名與句號。分子是符合條件的題數，分母是可驗證題數，不是「找到幾句 ÷ 全部句數」。

### Q30. 候選頁面涵蓋率和文件 Recall 有什麼關係？

涵蓋率把整個候選頁面庫當作找到的頁面；文件 Recall 只看其中排序前 N 頁。因為 top-N 一定是候選庫的子集，依這裡的同一套命中定義，Recall 不會高於候選頁面涵蓋率。前者偏低先改善候選範圍，涵蓋率高但 Recall 低則表示排序仍有改善空間。

### Q31. Classification Accuracy 跟本地 strict score 差在哪？

Accuracy 只看所有 dev 題目的標籤有沒有猜對，不要求證據正確。本地 strict score 對可驗證題同時要求標籤正確與前 5 句完整命中一組證據；NEI 只看標籤。本地計算是估計，正式 Kaggle 結果以競賽評分器為準。

### Q32. 驗證集 300 筆跟排行榜分數可以互換嗎？

不可以。Dev 指標是你用有標準答案的驗證資料自行計算；Kaggle public / private 分數是提交測試預測後，由競賽系統在不同部分評分。Report 需要分別填入，沒有實測時不應編造數字。若調整 dev 抽樣筆數，也要註明實際樣本大小。

### Q33. 為什麼分類器使用可能檢索錯誤的句子訓練？

程式直接把第一、二階段取回的 train context 給分類器，讓訓練輸入形式接近推論時情況；但取錯證據時，正確標籤與 context 可能不相符，形成較吵雜的訓練資料。可以研究使用金標證據或混合策略，但要說明資料來源與實際實驗，不能假裝已經比較過。

## 五、Colab 與交作業

### Q34. 為什麼建議我用 Colab，不在自己電腦跑？

這次需要大量 BERT 編碼與微調；使用 Colab GPU 可以讓你的電腦主要負責瀏覽與操作，運算在遠端執行。GPU 的可用性、執行時間及連線穩定性仍受 Colab 當下資源影響；請將重要輸出存到 Drive。

### Q35. ZIP 要上傳哪裡？我需要先解壓嗎？

在 Drive 建立 `MyDrive/nlp-hw3/`，上傳下載完成的競賽 ZIP；不用在電腦先解壓。Colab 掛載 Drive 後，程式會從 ZIP 取出四個需要的檔案到 `/content/cfever_data/`。若該資料夾有多個 ZIP，設定 `COLAB_ZIP_PATH` 指向正確一份。

### Q36. 看到 `找不到 ... ZIP` 或找不到 JSONL 怎麼辦？

先確認 Drive 已掛載、ZIP 上傳已完成、資料夾名稱及位置正確；程式預設找 `MyDrive/nlp-hw3/` 裡唯一的 ZIP。若手動改過 `COLAB_ZIP_PATH` 或 `DATA_DIR`，核對路徑。不要把 `.crdownload` 未完成檔當作完整 ZIP 使用。

### Q37. `CUDA out of memory` 該改哪個參數？

先嘗試把 `EMBEDDING_BATCH_SIZE` 從 16 調小；若發生在分類訓練階段，調小 `BATCH_SIZE`。仍不足可考慮降低對應最大長度，但內容被截斷可能影響表現。調整後從相關步驟重新執行，報告也要反映實際設定。

### Q38. 為什麼模型下載失敗，或離線 Kaggle 找不到模型？

Colab 第一次下載需要網路與足夠空間。Kaggle 的 `/kaggle/input` 是唯讀，離線重跑時需要預先把完整模型加入 Input，並在必要時設定 `MODEL_DIR` 及 `ALLOW_MODEL_DOWNLOAD = False`。單靠前一次互動執行的 `/kaggle/working` 不能保證下一次自動看得到。

### Q39. 跑完後要交 `metrics.json` 嗎？

`metrics.json` 是方便你複製真實驗證數字的紀錄；`submission.csv` 是給 Kaggle 產生排行榜分數的預測。課程要求提交的程式與報告在 notebook 內，檔名及壓縮格式請依課程規範確認；學習 Markdown 和 metrics 不會自動變成應繳的作業檔。

### Q40. Report 明明寫好了，為什麼還有「待填」？

方法描述與 AI 使用聲明已經先寫好；dev 數字需要實際執行，public 分數需要實際提交，private 分數需要等待公布。這些值無法從程式碼可靠推測。執行後第 5 節會印出 Q1～Q4 所需數字；把結果填進既有 `Answer:`，若改了設定也記得修正文字。

### Q41. 使用 ChatGPT 協助會不會違反作業規範？

本地提供的課程規範要求說明生成式 AI 的使用；程式開頭與 Report 已依目前使用情況寫入。你仍應自行閱讀、執行並核對內容，讓聲明反映實際使用範圍，且遵守課程對相似作業和引用來源的規定。若助教後續公告不同，以最新正式公告為準。

### Q42. `submission.csv` 有哪三欄？

`id` 對應 test ID 且維持原順序；`predicted_label` 是三類標籤之一；`predicted_evidence` 是 JSON 格式的 `[[頁名, 句號], ...]` 字串。程式在 NEI 時輸出 `[]`，並在存檔前檢查筆數、ID 順序與標籤集合。
