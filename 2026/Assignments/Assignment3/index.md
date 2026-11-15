# Assignment 3 入口：執行、讀懂、手打、繳交

這份是起點：先用這裡的步驟跑出真正的數字，再照「手打清單」把核心程式弄懂。整份 notebook 不用全部重抄。

> **v3 已經跑完，最後交 v3**：v1 是 0.776（第 64 名），v2 是 0.856（第 50 名），v3 是 **0.857**（dev FEVER 70.03%，比 v2 高 1.03 分）。v2 的漏斗顯示分數主要掉在「分類」，所以 v3 只把分類器換成 3 個模型的 ensemble。怎麼跑、改了什麼、跑完的分析和為什麼停在 v3，直接看最後的 **[九、v3](#九v3三個分類器的-ensemble)**；v2 的說明在 **[八、v2](#八v2微調-reranker--roberta-large)**。

---

## 一、檔案地圖

| 檔案 | 用途 | 要交嗎 |
| :--- | :--- | :---: |
| [st_minecode_v3.ipynb](st_minecode_v3.ipynb) | **v3 正式作業**：檢索沿用 v2，分類改成 3 個模型的 ensemble，Report 多了 v1 / v2 / v3 對照 | ✅ 交這份（要交的是跑完、Report 填好的那份，改名後交） |
| [st_minecode_study_v3.ipynb](st_minecode_study_v3.ipynb) | **v3 練功本**：v3 新增的 ensemble（§1～§8），加上 v3 跑完後的分析：交叉驗證、多頁證據（§9～§10）。10 節、6 個手打題，CPU 幾秒跑完，約 25 分鐘 | ❌ 自己看 |
| [st_minecode_v2.ipynb](st_minecode_v2.ipynb) | v2 正式作業：微調 reranker + RoBERTa-large，Report 有 v1 / v2 對照 | Kaggle 第 2 個 submission 選它當保險 |
| [st_minecode_study_v2.ipynb](st_minecode_study_v2.ipynb) | **v2 練功本**：只講 v2 新增的 10 個觀念，CPU 幾秒跑完 | ❌ 自己看 |
| [st_minecode.ipynb](st_minecode.ipynb) | v1 正式作業：程式 + 最下面的 Report。Kaggle、Colab 共用同一份 | v2 比較差才交這份 |
| [st_minecode_study.ipynb](st_minecode_study.ipynb) | **練功本**：10 個小積木，CPU 幾秒跑完，有 ✍️ 手打練習和自動對答案 | ❌ 自己看 |
| [st_question.md](st_question.md) | 「我大概會問的問題」：v1 是 Q1～Q43（含 Claude vs Gemini 建議的評估），v2 從 Q44、v3 從 Q59 開始 | ❌ 自己看 |
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
6. **斷線救援**：如果是在訓練分類器時斷線，重跑第 1 節、「讀取資料」、第 2 節第一格（`gold_groups` 等工具函式）和「用 bge-m3 把文字變成向量」那格，再把「Colab 斷線救援」那格的 `LOAD_RETRIEVAL_CACHE` 改成 `True`，從那格往下跑（見 [st_question Q38](st_question.md)）。v2 把 `resolve_model` 移進第 2 節第一格了，所以 v2 只要重跑前 3 種。

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

---

## 八、v2：微調 reranker + RoBERTa-large

v1 的 public score 是 **0.776**（第 64 名，第 59 名是 0.802）。FEVER score 要「證據找齊 + 標籤對」才拿分，所以 v2 **保留 v1 的文件檢索**，加強後面兩段：

| | v1 | v2 | 為什麼 |
| :--- | :--- | :--- | :--- |
| 初篩句數 | 20 | 30 | reranker 能挑到的上限變高 |
| reranker | 現成 | 用 train 的標準證據**微調 1 輪**（1 正例 + 7 個 hard negative，listwise loss）；dev 變差就自動改回現成的 | 現成模型沒學過 CFEVER 標註者挑哪一句 |
| 分類器 | `chinese-roberta-wwm-ext`（102M） | `chinese-roberta-wwm-ext-large`（326M） | 推論能力較強，仍 < 1B |
| 輸入 | 256 token、證據打亂 | 512 token、每句 ≤ 120 字、標準證據放前面 | v1 的標準證據可能被截掉（第 4 節會實際量出比例） |
| NEI 訓練證據 | 向量前 5 句 | reranker 從前 10 句挑 5 句 | 和 test 一樣難 |
| NEI 決策 | NEI bias | NEI bias ＋ reranker 門檻（一起在 dev 上挑） | 證據找不到時猜 S/R 也沒分 |
| epoch | 用最後一個 | 留 dev FEVER 最好的 | 避免過擬合 |

**作業規範**：3 個模型都 < 1B（568M / 568M / 326M，`count_parameters` 會檢查）；沒有外部 API；只用 CFEVER（train 訓練、dev 挑參數、test 只讀 claim）；Report 題目格式沒動。

### 怎麼跑（建議 Kaggle）

v2 在 T4 上大約要 **1.5～2 小時**，Colab 免費版容易中途斷線，所以建議用 Kaggle 的背景執行：

1. Kaggle **Code → New Notebook → File → Import Notebook**，上傳 `st_minecode_v2.ipynb`。
2. **Add Input** 加入競賽資料；**Settings**：Accelerator 選 **GPU T4 x2** 或 **P100**，**Internet 打開**。
3. 先按一次最上面兩格，確認印出「環境：Kaggle」和 GPU 名稱。
4. 右上角 **Save Version → Save & Run All (Commit) → Save**。它會在背景跑，關掉瀏覽器也沒關係（單次最多 12 小時，每週 GPU 30 小時）。
5. 跑完後到這個 notebook 的頁面 → 該版本 → **Output**：下載 `submission.csv` 提交，`report_filled.md` 就是填好的報告。

**Colab 也可以**：步驟和 v1 一樣，輸出會存在 Drive 的 `nlp-hw3/output_v2/`（不會蓋掉 v1）。如果 v1 的 `nlp-hw3/output/metrics.json` 還在，第 7 節會**自動讀進來做 v1 / v2 對照表**。斷線時：重跑第 1 節、「讀取資料」、第 2 節第一格，再把 `LOAD_RETRIEVAL_CACHE` 改成 `True`，從那格往下跑。

| 步驟（T4，估計值） | 大約時間 |
| :--- | :--- |
| 下載 3 個模型（多了 1.3 GB 的 large） | 3～6 分鐘 |
| 讀維基 + 標題比對 + bge-m3 編碼（同 v1） | 20～25 分鐘 |
| 現成 reranker 重排 dev | 3～5 分鐘 |
| 微調 reranker（8,000 題） | 15～25 分鐘 |
| 重排 dev、test、train 的 NEI 題 | 10～15 分鐘 |
| 訓練 large 分類器 2 epochs + 預測 | 30～40 分鐘 |
| **合計** | **約 1.5～2 小時** |

**出問題時**：

- `CUDA out of memory` 在分類器：`BATCH_SIZE = 4`、`GRAD_ACCUM_STEPS = 4`（等效 batch 不變），還不夠就 `GRADIENT_CHECKPOINTING = True`。
- `CUDA out of memory` 在微調 reranker：`RERANK_TRAIN_GROUPS = 2`。
- 時間不夠：`FINETUNE_RERANKER = False`（省 20 分鐘，報告第 2 題 (3) 要自己刪掉），或 `EPOCHS = 1`。
- dev Accuracy 卡在 40% 左右（large 偶爾學不起來）：`LEARNING_RATE = 5e-6` 或換 `SEED` 重跑。

### v2 新增的格子

| 節 | 格子 | 做什麼 | 手打？ |
| :--- | :--- | :--- | :---: |
| 3 | 向量初篩 30 句 | train 也取 30 句，給微調找 hard negative；用完釋放句子向量 | |
| 3 | 現成 reranker | `load_reranker`（fp32 載入）、`rerank` 多回傳第 1 名分數 | |
| 3 | 微調資料 | 每題 1 正例 + 7 個 hard negative（從初篩前 14 名、排除標準證據） | ✍️ |
| 3 | `finetune_reranker` | 凍結詞向量、`view(題數, 8)` + cross entropy、fp16；dev 比較後選一個 | ✍️ |
| 3 | 處理 test 與 train NEI | 選好的 reranker 重排 test；train 的 NEI 題從前 10 句挑 5 句 | |
| 3 | `format_sentence` / `build_context` | 每句 ≤ 120 字；S/R 的標準證據放前面、不打亂 | ✍️ |
| 4 | 截斷分析 | `gold_kept_ratio`：v1 設定 vs v2 設定，標準證據留下來的比例 | |
| 4 | Training | gradient accumulation、保留 dev FEVER 最好的 epoch | ✍️ |
| 5 | `decide_labels` | NEI bias × reranker 門檻的 grid search | ✍️ |
| 7 | 報告 | 多了 v1 / v2 對照表、截斷分析、門檻表 | |

### v2 手打清單（練功本 [st_minecode_study_v2.ipynb](st_minecode_study_v2.ipynb)）

| 順序 | 練功本 | 打完應該能回答 |
| :---: | :--- | :--- |
| 1 | §2 `build_context` | 為什麼標準證據放前面比打亂好？`sentence[:120]` 在句子比較短時會怎樣？ |
| 2 | §3 `make_group` | 什麼是 hard negative？為什麼一定要先排除標準證據？ |
| 3 | §4 `listwise_loss` | `[題數 × 8, 1]` 怎麼變成 `[題數, 8]`？target 為什麼全是 0？ |
| 4 | §5 `freeze_embeddings` | 凍結詞向量省了多少記憶體？為什麼不會讓 reranker 變笨？ |
| 5 | §7 `accumulated_grad` | loss 為什麼要除以累積次數？什麼時候才 `step` 和 `zero_grad`？ |
| 6 | §8 `pick_best` | 為什麼存權重要 `.clone()`？最後一個 epoch 為什麼不用另外存？ |
| 7 | §9 `decide_labels` | 門檻為什麼用分位數？為什麼「找不到證據就判 NEI」不會讓 FEVER 變差？ |

### 提交與繳交

- v2 的 `submission.csv` 交上 Kaggle 後，把 public score 填進 v2 Report 的 `Your public leaderboard score:`。
- 截止（11/23 12:00）前選最終 2 個 submission：建議 **v2 + v1**（或 v2 + 另一個 v2 的設定），一份分數最高、一份做保險。
- NTU COOL 只交一份：**分數比較好的那一版**改名成 `NLP_HW3_NTOU_11457088.ipynb`。交 v2 的話，Report 裡已經有 v1 / v2 對照表，第 4 題會比較完整。

---

## 九、v3：三個分類器的 ensemble

v2 的 public score 是 **0.856**（第 50 名），dev FEVER score 69.00%。v2 的三階段漏斗（dev 3,000 題）把分數掉在哪裡算得很清楚：

| 掉在哪一步 | 題數 |
| :--- | ---: |
| ① 文件檢索：前 5 頁沒找齊證據頁 | 223 |
| ② 證據檢索：頁面對了，前 5 句沒找齊 | 60 |
| ③ **分類：證據已經找齊，標籤卻判錯** | **417** |
| NOT ENOUGH INFO 判錯 | 230 |

分類掉的比前兩步加起來還多，所以 v3 **檢索完全沿用 v2，只改分類**：

| | v2 | v3 | 為什麼 |
| :--- | :--- | :--- | :--- |
| 分類器 | `chinese-roberta-wwm-ext-large` | 再加 `chinese-macbert-large`、`xlm-roberta-large`，三類機率平均 | 預訓練方式、斷詞方式不同，錯的題目不完全一樣 |
| 模型組合 | — | 7 種組合（單一、兩兩、全部）都在 dev 上比，選 FEVER 最高的 | 某個模型拖後腿時會自動不用它 |
| NEI bias 範圍 | -1.5～1.5 | -3.0～1.0 | v2 選到邊界的 -1.5，更小的值可能更好 |
| 門檻分位數 | 每 5% 一格 | 10% 附近每 2.5% 一格 | v2 選到 10%，在附近找細一點 |
| 斷線 | 分類器要整個重訓 | 每個模型訓練完就存 `pred_*.json`，重跑時自動跳過 | 3 個模型多花 1 個多小時 |

**沒改的**：文件檢索、證據檢索、輸入格式。輸入格式我用 v2 的檢索結果檢查過：dev 上證據有找齊的 1,717 題裡，1,514 題（88%）的標準證據排在 reranker 的第 1 句，和訓練時「標準證據放最前面」一致，所以不用改。

**作業規範**：5 個模型每個都 < 1B（568M / 568M / 326M / 326M / 560M），1B 是逐一計算（11/06 答覆）；3 個分類器都只做過語言模型預訓練，理由和 v2 不用 Erlangshen-NLI 一樣（見 [st_question Q61](st_question.md)）；沒有外部 API；只用 CFEVER。

### 怎麼跑

**方法 A：Colab 沿用 v2 的檢索結果（推薦，L4 約 1.5～2 小時）**

1. 確認 Drive 的 `nlp-hw3/output_v2/retrieval_cache.json` 還在（v2 跑完就有）。
2. Colab 開啟 `st_minecode_v3.ipynb`，執行階段類型選 GPU（v2 用的是 L4）。
3. 依序執行第 1 節的三格（套件、環境、參數）、「讀取資料」那格、第 2 節第一格（`gold_groups` 那格）。
4. 捲到第 3 節最後「沿用 v2 的檢索結果」那格，把 `LOAD_RETRIEVAL_CACHE` 改成 `True`，按 Ctrl+Enter，確認印出「已載入檢索快取：…output_v2/retrieval_cache.json」。
5. 點下一格（第 4 節的標題），**執行階段 → 執行之後的儲存格**（Ctrl+F10）。
6. 輸出在 Drive 的 `nlp-hw3/output_v3/`：`submission.csv`、`report_filled.md`、`metrics.json`，還有 3 個 `pred_*.json`。

**方法 B：完整重跑**：Run all。Colab L4 約 2.5 小時；Kaggle T4 約 4～5 小時（Save & Run All 背景執行）。檢索會重跑一次，數字可能和 v2 有一點點差異（reranker 的微調在 GPU 上不是完全可重現）。

> Kaggle 的 Save & Run All 只能整份跑，沒辦法跳過檢索，所以 Kaggle 就用方法 B。想在 Kaggle 用快取，要開互動模式照方法 A 的步驟手動跑，並把 v2 的 `output_v2` 資料夾上傳成 Dataset 再 Add Input。

| 步驟（Colab L4，估計值） | 方法 A | 方法 B |
| :--- | :--- | :--- |
| 檢索（同 v2） | 不用跑，讀快取 | 約 40 分鐘 |
| roberta-large 2 epochs + 預測（v2 實測一個 epoch 15 分鐘） | 約 35 分鐘 | 約 35 分鐘 |
| macbert-large（同大小） | 約 35 分鐘 | 約 35 分鐘 |
| xlm-roberta-large | 約 35 分鐘 | 約 35 分鐘 |
| 模型組合比較 + 報告 | 1～2 分鐘 | 1～2 分鐘 |
| **合計** | **約 1.5～2 小時** | **約 2.5 小時** |

**出問題時**：

- **中途斷線**：訓練完的模型已經存在 `output_v3/pred_*.json`。照方法 A 的步驟 3～5 重來，已經訓練好的模型會印出「讀取已經訓練好的 …」直接讀檔，只會重訓還沒完成的那個。
- **`CUDA out of memory`**：`BATCH_SIZE = 4`、`GRAD_ACCUM_STEPS = 4`（等效 batch 不變），還不夠就 `GRADIENT_CHECKPOINTING = True`。
- **印出「⚠️ … 沒有學起來，不放進 ensemble」**：最可能是 xlm-roberta-large。ensemble 會自動只用其他模型，不處理也能交；想救的話把它的 `lr` 改成 `5e-6`，從第 4 節重跑（設定變了會自動重訓，其他模型直接讀檔）。
- **時間不夠**：把 `CLASSIFIER_MODELS` 刪到剩 2 個。

### v3 改了哪些格子

| 節 | 格子 | 改了什麼 | 手打？ |
| :---: | :--- | :--- | :---: |
| 1 | 參數 | `CLASSIFIER_MODELS`（每個模型的 lr、要不要凍結詞向量）、`REUSE_SAVED_PREDICTIONS`、放寬 `NEI_BIAS_GRID` / `SCORE_QUANTILES`、`PREVIOUS_RESULTS`（v1 / v2 的實際數字） | |
| 3 | 沿用 v2 檢索結果 | `find_retrieval_cache`：先找 `output_v3/`，再找 `output_v2/` 或 Kaggle Input | |
| 4 | `ClaimDataset` / `make_loaders` | tokenizer 變成參數，每個模型重建 DataLoader；印出 3 個 tokenizer 切出來的長度 | |
| 4 | `train_classifier` | v2 的訓練迴圈包成函式；`freeze_embeddings`；回傳 logits 後刪掉模型 | 🟡 |
| 4 | 依序訓練 | `pred_*.json` 存檔 + 輸入指紋；沒學起來的模型自動排除 | |
| 5 | `ensemble_scores` | softmax → 平均 → 取 log | ✍️ |
| 5 | 模型組合 | `combinations` 列出所有組合，各自挑 bias / 門檻，選 dev FEVER 最高的 | ✍️ |
| 5 | 漏斗 | 多印「標準證據排在第 1 句」的比例 | |
| 7 | 報告 | v1 / v2 / v3 對照表、每個模型的表、模型組合表 | |

### v3 手打清單

v3 的新程式在正式 notebook 裡要先訓練完 3 個模型（1.5～2 小時）才跑得到，沒辦法改一行就跑一下，所以還是做了一本小的練功本 [st_minecode_study_v3.ipynb](st_minecode_study_v3.ipynb)（用假的 logits，CPU 幾秒跑完，約 25 分鐘）。先在練功本打，再回正式 notebook 對照：

| 順序 | 練功本 | 正式 notebook | 打完應該能回答 |
| :---: | :--- | :--- | :--- |
| 1 | §3 `ensemble_scores` | 第 5 節 `ensemble_scores` | 為什麼平均「機率」而不是平均 logits（§2）？最後為什麼要取 log？`clamp_min(1e-9)` 在防什麼？ |
| 2 | §4 `pick_best_combo` | 第 5 節的 `combinations` 迴圈 | 3 個模型有幾種組合？同分時為什麼選模型多的？ |
| 3 | §5 `same_setting` | 第 4 節 `input_fingerprint` / `same_setting` | 什麼情況會沿用 `pred_*.json`、什麼情況會重訓？為什麼用 `.get`？ |
| 4 | §8 `standard_error` | （決定要不要提交 v3） | v2 0.856 vs v3 0.857 算不算真的進步？為什麼以 dev 為準？ |
| 5 | §9 `kfold_indices` | （v3 跑完後的分析，正式 notebook 沒有） | 在 dev 上調參、又在 dev 上報分數，會高估多少？為什麼 v3 沒改用 stacker？ |
| 6 | §10 `easiest_group` | （v3 跑完後的分析，正式 notebook 沒有） | 為什麼要 2 頁以上才能證明的題目特別難找齊？第二輪檢索在補什麼？ |

看懂就好的：§1 ensemble 為什麼有用（含 v3 實測：3 個模型 73.5% 的題目意見一樣）、§6 學不起來的模型自動排除、§7 為什麼只凍結 xlm-roberta 的詞向量（`trainable` 為什麼同時傳給 optimizer 和梯度裁剪）。

### v3 跑完的結果

| | v2 | v3 |
| :--- | :---: | :---: |
| dev Evidence Recall@5 | 85.85% | 85.50%（重跑 reranker 微調，有一點差異） |
| dev Accuracy | 73.30% | 74.77% |
| dev FEVER score | 69.00% | **70.03%** |
| Kaggle public | 0.856 | **0.857** |

- dev 多對 31 題，public 只差約 1 題（public 的誤差約 ±22 題，見練功本 §8），所以以 dev 為準：**v3 比 v2 好**。
- 分類這一步少錯 55 題（417 → 362），但證據沒找齊的 290 題沒有改善，其中 179 題要 2 頁以上才能證明（練功本 §10）。
- 想再往上的評估：NEI 判斷改用 stacker，交叉驗證後只多 0.23 分，是雜訊（練功本 §9）。第二輪檢索可以讓頁面涵蓋率從 88.85% 升到 91.90%～93.10%，但估計 public 只到 0.87～0.88，還要多跑 3～4 小時 GPU，到 0.90 機會不大，所以**停在 v3**。

### 提交與繳交

- v3 的 dev FEVER 70.03% 比 v2 的 69.00% 高，已經提交（`submission_v3.csv`，public 0.857）。
- Kaggle public 只用約 17% 的 test（約 1,000 題），差 1 分以內都可能是運氣；v3 有沒有比 v2 好，以 dev（3,000 題）為準。
- 截止（11/23 12:00）前選最終 2 個 submission：**v3 + v2**，一份分數最高、一份做保險。GPT 版的 0.551 不要選。
- NTU COOL 只交一份：v3。Colab 跑完下載的 notebook 已經把 Report 填好、public 填了 0.85700，存成 `Downloads/NLP_HW3_NTOU_11457088.ipynb`。
- 11/23 private 公布後，把 `Your private leaderboard score:` 填上，再壓成 `NLP_HW3_NTOU_11457088.zip`（zip 裡只放這個 ipynb）。學校代碼和學號交之前再核對一次。
