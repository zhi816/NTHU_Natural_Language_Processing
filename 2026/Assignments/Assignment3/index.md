# Assignment 3 入口：執行、讀懂、手打、繳交

這份是起點：先用這裡的步驟跑出真正的數字，再照「手打清單」把核心程式弄懂。整份 notebook 不用全部重抄。

---

## 一、檔案地圖

| 檔案 | 用途 | 要交嗎 |
| :--- | :--- | :---: |
| [st_minecode.ipynb](st_minecode.ipynb) | **正式作業**：程式 + 最下面的 Report。Kaggle、Colab 共用同一份 | ✅（改名後交） |
| [st_minecode_study.ipynb](st_minecode_study.ipynb) | **練功本**：10 個小積木，CPU 幾秒跑完，有 ✍️ 手打練習和自動對答案 | ❌ 自己看 |
| [st_question.md](st_question.md) | 43 個「我大概會問的問題」，包含 Claude vs Gemini 建議的評估 | ❌ 自己看 |
| [作業規範.md](作業規範.md)、[作業補充.md](作業補充.md) | 配分、規定、助教答疑 | ❌ |
| `st_code.ipynb`、`st_code_kaggle.ipynb`、`程式逐段解析.md`、`issue.md` | 之前 GPT 做的版本（含 Kaggle 版）與審查，留著參考，**不用再看**。舊的 index / st_question 在 git 紀錄 `ac424a8` 裡 | ❌ |
| `sample_code.ipynb` | 助教原始 sample code | ❌ |

### 為什麼「程式分析」做成練功本 notebook，而不是一份 .md？

我原本也在想要寫一份逐行解析的 .md，最後選了 notebook，原因有三個：

1. **我是邊跑邊懂的人**。HW2 的 `my_main.ipynb` 就是在程式旁邊一段一段寫註解、改一改再跑。.md 只能看，不能跑。
2. **正式 notebook 不適合拿來實驗**。它要讀 1 GB 維基、下載 3 個模型、跑 40～60 分鐘 GPU；想看「mean pooling 到底在算什麼」不可能每次都跑一遍。練功本只用幾行假資料，CPU 幾秒鐘就跑完。
3. **手打要有回饋**。練功本的練習格子會自動對答案（⏳ / ✅ / ❌），打錯馬上知道；參考答案用 `<details>` 收起來，先自己寫再點開。

分工是：**正式 notebook** 每一節前面的 markdown 講「這格在做什麼、為什麼」；**練功本**講「這幾行到底怎麼運作」；**st_question.md** 回答「為什麼這樣選、遇到錯誤怎麼辦」；**這份 index.md** 是地圖。

### 為什麼 Kaggle 和 Colab 只有一份 notebook？

兩個平台只差「資料放哪、輸出存哪」，大約 15 行，第 1 節第二格會自動判斷環境。分成兩份的話，改 bug 要改兩次、報告數字可能對不起來，而且 NTU COOL 本來就只收一份 `.ipynb`。

---

## 二、30 秒看懂整個流程

```text
                         ┌─ 第 1 次讀維基：只讀頁名 → 「標題 → 頁面」索引（一百多萬頁）
claim ─► ① 文件檢索 ─────┤  claim 的每個子字串去查表 → 標題命中頁
                         └─ 第 2 次讀維基：只載入候選池（train 標註頁 + 標題命中頁）
                            bge-m3 向量 cosine + 標題命中加分（bonus 在 dev 上挑）→ 前 5 頁
      ─► ② 證據檢索 ─────  5 頁的「每一句」（前面加頁名）
                            bge-m3 向量初篩 20 句 → bge-reranker 重排 → 前 5 句（保留原始句號）
      ─► ③ 分類 ──────────  [CLS] claim [SEP] 【頁名】句子 ×5 [SEP] → chinese-roberta-wwm-ext
                            訓練：S/R 補進標準證據、NEI 用檢索結果；預測：NEI logit 加 bias（在 dev 上挑）
      ─► submission.csv ＋ 第 7 節自動印出填好數字的 Report
```

| 模型 | 用途 | 參數量 |
| :--- | :--- | ---: |
| `BAAI/bge-m3` | 頁面、句子、claim 的向量 | 約 568M |
| `BAAI/bge-reranker-v2-m3` | 句子重排（cross-encoder） | 約 568M |
| `hfl/chinese-roberta-wwm-ext` | 三分類 | 約 102M |

每個都 < 1B，程式裡的 `count_parameters` 會再檢查一次。全部在本地 GPU 跑，沒有呼叫任何外部 API；訓練只用 CFEVER。

---

## 三、跑一次

### 方法 A：Kaggle（推薦，最後也要在這裡交）

1. 到[競賽頁面](https://www.kaggle.com/competitions/nthu-nlp-2026-assignment-3-fact-verification) → **Code → New Notebook**，選 **File → Import Notebook** 上傳 `st_minecode.ipynb`。
2. 右側 **Add Input** 加入這個競賽的資料。
3. 右側 **Settings**：**Accelerator 選 GPU**（T4 或 P100），**Internet 打開**（第一次要下載 3 個模型，約 5 GB）。
4. **Run All**。第 1 節會印出「環境：Kaggle」和 4 個資料檔的路徑，確認沒有報錯就可以先去做別的事。
5. 跑完後，右側 **Output** 會有 `submission.csv`、`report_filled.md`、`metrics.json`。用 `submission.csv` 提交（或 Save Version 之後從 Output 提交）。
6. 每次提交都在 notebook 或筆記裡記下設定，截止前要自己**選最多 2 個 submission**。

> 如果競賽規定提交時不能開 Internet：先在有網路的情況下跑一次，把 3 個模型存成 Kaggle Dataset 再加入 Input。資料夾名稱要包含 `bge-m3`、`bge-reranker-v2-m3`、`chinese-roberta-wwm-ext`，`resolve_model` 會自動找到。

### 方法 B：Colab

1. 在 Google Drive「我的雲端硬碟」建立 `nlp-hw3` 資料夾，把競賽 ZIP 放進去（資料夾裡只能有這一個 `.zip`）。已經解壓的 4 個 `.jsonl` 直接放進去也可以。
2. Colab 開啟 `st_minecode.ipynb`。notebook 已經預設 GPU，還是可以到 **執行階段 → 變更執行階段類型** 確認是 T4 GPU。
3. **執行階段 → 全部執行**，跳出 Drive 授權時同意。
4. 輸出會存在 Drive 的 `nlp-hw3/output/`：`submission.csv`、`report_filled.md`、`metrics.json`、`retrieval_cache.json`、`classifier/`。
5. 把 `submission.csv` 下載下來，到 Kaggle 競賽頁 **Submit Prediction** 上傳。
6. **斷線救援**：如果是在訓練分類器時斷線，重跑第 1 節（到「讀取資料」為止），把「Colab 斷線救援」那格的 `LOAD_RETRIEVAL_CACHE` 改成 `True`，再從那格往下跑（見 [st_question Q38](st_question.md)）。

### 大概要多久（T4 GPU，估計值）

| 步驟 | 大約時間 | 用什麼 |
| :--- | :--- | :--- |
| 下載 3 個模型 | 2～5 分鐘 | 網路 |
| 讀維基兩次 + 標題比對 | 3～5 分鐘 | CPU |
| bge-m3 編碼頁面、claim、句子 | 15～20 分鐘 | GPU |
| reranker 重排 dev + test | 5～10 分鐘 | GPU |
| 訓練分類器 2 epochs + 預測 | 10～15 分鐘 | GPU |
| **合計** | **約 40～60 分鐘** | |

這些是估計值，實際時間會因為 GPU 型號、候選池大小而變。想先確認流程不會出錯，可以照 [st_question Q40](st_question.md) 設定試跑。

---

## 四、正式 notebook 目錄（每一格在做什麼）

| 節 | 格子（標題 / 函式） | 做什麼 | 重要的變數 / 輸出 | 手打？ |
| :--- | :--- | :--- | :--- | :---: |
| 1 | 匯入套件、`set_seed` | 固定亂數種子、決定 GPU / CPU、程式開頭的 AI 使用註解 | `device`、`USE_FP16` | |
| 1 | 自動判斷環境 | Kaggle / Colab / 本機各自的資料與輸出路徑；Colab 只解壓需要的 4 個檔 | `DATA_DIR`、`OUTPUT_DIR` | |
| 1 | 參數 | 資料量、模型名稱、top-k、bonus / bias 的候選值、訓練超參數 | `TRAIN_SAMPLES`…`NEI_BIAS_GRID` | |
| 1 | 讀取資料 | 讀 JSONL、標籤轉大寫、抽樣、`reset_index` | `train_df`、`dev_df`、`test_df` | |
| 2 | 工具函式 | `gold_groups`、`title_key`、`doc_hit`、`evidence_hit`、`recall_at`、`md_table` | `report_values` | 🟡 |
| 2 | 第 1 次讀維基 | 只讀頁名，建「標題 → 頁面」索引；跳過消歧義頁 | `title_index` | |
| 2 | 標題比對 | `match_titles`：claim 子字串查表，長標題優先 | `*_title_matches` | ✍️ |
| 2 | 第 2 次讀維基 | 只載入候選池的開頭文字和句子；`parse_lines` 保留原始句號 | `page_lead`、`page_sentences` | |
| 2 | 載入 bge-m3 | `resolve_model`、`count_parameters`、`encode_texts`（排序分批、CLS pooling、L2 normalize） | `retriever` | ✍️ |
| 2 | 編碼 | 頁面「頁名：開頭 200 字」與所有 claim 變成向量 | `page_embeddings`、`*_claim_emb` | |
| 2 | 混合分數 | `page_candidates`（分批矩陣乘法 + topk）、`rank_pages`、在 dev 上挑 bonus、Recall@1/3/5/10 | `best_bonus`、`*_docs` | ✍️ |
| 3 | 句子索引 + 編碼 | 前 5 頁的每一句加頁名後編碼；編完釋放 bge-m3 | `sentence_keys`、`page_sentence_rows` | |
| 3 | 向量初篩 | `rank_sentences_dense`：local → 全域 index | `dev_dense_rows` | ✍️ |
| 3 | reranker | `rerank`：dev / test 的 20 句做 cross-encoder 重排 | `dev_evidence`、`test_evidence` | |
| 3 | Evidence Recall | 初篩 vs 重排 vs 上限；「文件已找對」時的證據命中率 | `evidence_table` | |
| 3 | 準備分類輸入 | `build_context`；S/R 補進標準證據、打亂；存 `retrieval_cache.json` | `train_contexts` | ✍️ |
| 3 | 斷線救援 | `LOAD_RETRIEVAL_CACHE = True` 時從快取恢復 | | |
| 4 | Dataset / DataLoader | `ClaimDataset`（`only_second` 截斷）、dynamic padding | `train_loader` | |
| 4 | 模型與 optimizer | 三分類頭、AdamW、warmup 排程、GradScaler；`predict_logits`、`fever_score` | `classifier` | 🟡 |
| 4 | Training | 5 步驟 + fp16；每個 epoch 在 dev 上評估 | `epoch_table` | ✍️ |
| 5 | Validation | NEI bias 表、Accuracy、FEVER、混淆矩陣、三階段漏斗 | `best_bias`、`funnel_table` | |
| 5 | 錯誤分析 | 各印 3 題「文件沒找到 / 句子沒找齊 / 標籤錯」 | | |
| 6 | submission | 三欄、NEI 填 `[]`、順序檢查 | `submission.csv` | |
| 7 | 報告自動填空 | 依數字產生分析句，把模板填好印出，並存 `report_filled.md`、`metrics.json` | `report_text` | |
| — | Report | 原始題目格式不變；跑完後用第 7 節的輸出整段取代 | | |

---

## 五、手打清單（依重要性）

先在練功本手打，打對了再回正式 notebook 對照同一段。每段打完都應該能回答右邊的問題。

| 順序 | 練功本 | 正式 notebook | 打完應該能回答 |
| :---: | :--- | :--- | :--- |
| 1 | §2 `match_titles` | 第 2 節「標題比對」 | 為什麼是「子字串去查表」而不是「標題去 claim 裡找」？`.get(key, [])` 為什麼不會讓 dict 變大？ |
| 2 | §3 mean pooling + normalize | `encode_texts` | `unsqueeze(-1)` 之後形狀是什麼？為什麼 normalize 後內積 = cosine？bge-m3 為什麼取 [CLS]？ |
| 3 | §4 `rank_pages` | `page_candidates` / `rank_pages` | 相似度矩陣一列代表什麼？bonus = 0 和 bonus = 1 的差別？ |
| 4 | §5 `top_sentences` | `rank_sentences_dense` | `local` 和 `rows[local]` 差在哪？寫錯會不會報錯？ |
| 5 | §7 `inject_gold` | 「準備分類器的輸入文字」 | 為什麼補標準證據不算作弊？為什麼要打亂？ |
| 6 | §8 訓練 5 步驟 | Training | 哪一行真的更新參數？GradScaler 為什麼要先 `unscale_` 才 clip？ |
| 7 | §9 `fever_score` | `fever_score`、NEI bias | 為什麼 Accuracy 對、FEVER 卻可能是 0？bias 往上調會發生什麼事？ |

**看懂就好，不用手打**：環境判斷、Colab 解壓、`resolve_model`、`ClaimDataset`、submission、報告填空。這些都是固定寫法，和作業的核心觀念無關。

---

## 六、報告的數字從哪裡來？

第 7 節會把下面全部自動填好，**整段貼到最下面的 Report 格子就好**（題目和格式沒有改）。

| Report | 主要數字 | 附的表 |
| :--- | :--- | :--- |
| Q1 文件檢索 | Document Recall@5、命中題數 / 可驗證題數、選到的 bonus、候選涵蓋率 | 三種方法 × Recall@1/3/5/10 |
| Q2 證據檢索 | Evidence Recall@5、文件已找對時的證據命中率 | 初篩 vs 上限 vs 重排 |
| Q3 分類 | Accuracy（加 bias 與不加）、訓練筆數、補進標準證據的筆數 | 每個 epoch 的 loss / Acc / FEVER、混淆矩陣 |
| Q4 加分題 | 三階段漏斗、bonus 消融、NEI bias 取捨、規定檢查、限制與未來方向 | 漏斗、bonus 表、bias 表 |
| Q5 AI 聲明 | 已寫好，**請自己確認內容和實際使用情況一致** | |
| public / private | 要自己去 Kaggle 看，第 7 節沒辦法幫你填 | |

---

## 七、繳交前檢查清單

- [ ] 在 Kaggle（或 Colab）用**完整資料**跑完（`TRAIN_SAMPLES` / `DEV_SAMPLES` / `TEST_SAMPLES` 都是 `None`），每一格都有輸出。
- [ ] 第 7 節印出的報告已經貼到 Report 格子，沒有殘留【待填】。
- [ ] `Your Kaggle username:` 和 **leaderboard 上顯示的名字**一樣（目前填 `zhi816`，請到排行榜確認）。沒填或填錯扣 10 分。
- [ ] public score 已填；private score 在 11/23 12:00 公布後補上。
- [ ] Q5 的 AI 使用聲明和實際情況一致；程式最上方的 AI 註解還在。
- [ ] Kaggle 截止（11/23 12:00）前，已經**自己選好最多 2 個 submission**。
- [ ] 下載保留輸出的 notebook，改名成 **`NLP_HW3_NTOU_11457088.ipynb`**，壓成 **`NLP_HW3_NTOU_11457088.zip`**（學校、學號照 HW2，交之前再核對一次）。
- [ ] 11/25（三）23:59 前上傳 NTU COOL。
