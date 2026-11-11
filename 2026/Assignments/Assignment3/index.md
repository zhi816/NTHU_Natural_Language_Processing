# Assignment 3：執行與手打核心指南

這份是你的入口。先用 Colab 跑出真正的數據，再讀最值得親手打的程式；不用把整份 notebook 重新抄一遍。

## 要打開哪個檔案？

| 檔案 | 何時看 | 內容 |
| --- | --- | --- |
| [st_code.ipynb](st_code.ipynb) | 要執行、產生 CSV、寫報告 | 作業程式與最下方 Report；題目文字保留原樣 |
| [程式逐段解析.md](程式逐段解析.md) | 看不懂某個 cell | 按執行順序解釋每個程式區塊、資料型態、向量形狀和評估 |
| [st_question.md](st_question.md) | 卡在名詞、Python 語法或 Colab 錯誤 | 模擬這次可能會問的問題，以小例子說明 |
| [作業規範.md](作業規範.md) | 確認課程限制 | 配分、繳交格式與助教補充 |

我選擇用一份 Markdown 做完整程式解析，而不是複製另一份 notebook：程式只維護 `st_code.ipynb`，文字可以依函式名稱快速查找，也比較不會因為兩份程式改得不同而混淆。需要動手練習時，可以在 Colab 另外新增暫時的 cell，參照本文手打；不要讓練習 cell 改掉正式使用的變數。

## 先抓住整體流程

```text
train 的證據頁名 ──► 候選 Wikipedia 頁面 ──► BERT 頁面向量
                                                        │
claim ──► BERT claim 向量 ──► 找最相似的 5 頁 ──► 在這些頁面找最相似的 5 句
                                                           │
                           claim + 5 句 ──► 訓練好的 BERT 分類器 ──► 三個標籤
                                                           │
                                        test 的預測 ──► submission.csv
```

- 文件庫只包含「訓練資料曾標註過」的頁面，因此它不是完整 Wikipedia 檢索；這個限制會影響結果。
- 第二階段會考慮取回頁面的非空句子，不限制只能選曾被訓練資料標註的句子。
- `test.jsonl` 只使用 `id` 和 `claim`。測試 claim 會正常進行檢索，但不使用測試標準答案。
- 數字要等真實執行或 Kaggle 評分後才能填；`zhi816` 已填在 Report，分數不會預先猜測。

## Colab 一次跑通

你下載完成的檔案是 `nthu-nlp-2026-assignment-3-fact-verification.zip`，約 223 MiB。ZIP 內包含四個需要的 JSONL 及官方 `sample_submission.csv`；其中 `wiki_pages.jsonl` 解壓後約 1 GB。不需要先在自己的電腦解壓或跑模型。

1. 在 Google Drive 的「我的雲端硬碟」建立 `nlp-hw3` 資料夾，將上面的競賽 ZIP 上傳到這個資料夾。先等待上傳完成。預設該資料夾內只能有一份 `.zip`；若有多份，在 notebook 的 `COLAB_ZIP_PATH` 指定這次資料 ZIP 的完整路徑。
2. 到 Colab 開啟／上傳本資料夾的 `st_code.ipynb`。選單中選「執行階段 → 變更執行階段類型 → GPU」，儲存後確認第一個程式 cell 顯示 `cuda`。若顯示 `cpu`，先檢查 GPU 是否可用；CPU 也能跑，但可能非常慢。
3. 由上往下執行。Drive 掛載時依畫面授權。程式會從 `MyDrive/nlp-hw3/` 的 ZIP 把四個 JSONL 解壓至 Colab 的 `/content/cfever_data/`，並將結果存到 Drive 的 `MyDrive/nlp-hw3/output/`。
4. 第一次會下載中文 BERT 權重，模型檔本身約 412 MB；微調後模型也會存到 Drive。請預留空間和時間，並留意 Colab 可能斷線。句子編碼會印出進度；每次換新的 Colab 執行環境，暫存的資料通常需重新解壓。
5. 預設 `TRAIN_SAMPLES = 3000`、`DEV_SAMPLES = 300`、`EPOCHS = 1`。可以先降低 `TRAIN_SAMPLES` / `DEV_SAMPLES` 確認流程，再改回去從上方重跑；不要把小樣本結果說成完整資料結果。實際耗時與 GPU、候選句數和連線有關，沒有保證的固定分鐘數。
6. 完整執行後，到 Drive 的 `nlp-hw3/output/` 找 `metrics.json`、`submission.csv`、`finetuned_bert/` 與下載的模型資料夾。`metrics.json` 是你那次執行的 dev 結果；`submission.csv` 是 test 預測，不是分數。
7. 參照 notebook 第 5 節印出的 Q1～Q4 數值，把實測結果填入最下方 Report 既有的 `Answer:`；不要修改題目或 Markdown 格式。把 `submission.csv` 依競賽頁面的指示提交，取得 public score 後再填寫；private score 要等公布後才能填。
8. 檢查 Report 裡的 AI 使用聲明是否與自己的實際使用一致。下載保留執行結果的 notebook，依規範改名為 `NLP_HW3_<學校>_<學號>.ipynb`，再依規範壓成同名 ZIP 上傳；學校和學號請自行核對，不要直接照抄範例。

Kaggle 執行時，競賽資料要先加入 Input；程式會搜尋 Input。首次抓模型需 Internet，離線提交需要先將完整 `bert-base-chinese` 模型加入 Input 並設定 `ALLOW_MODEL_DOWNLOAD = False`。Kaggle 的 `/kaggle/working` 輸出不會自動變成下次執行的 Input。

## 最值得手打的 4 小段

| 順序 | 程式位置 | 為什麼親手打 | 打完應能回答 |
| --- | --- | --- | --- |
| 1 | `encode_texts` 的 `mask`、加總、平均與 `F.normalize` | 文字如何變成能比較的固定長度向量 | 為何 padding 不應參加平均？正規化後內積為何可比 cosine？ |
| 2 | `retrieve_documents` 的矩陣乘法與 `topk` | 理解第一階段真的如何排序 | 一列是什麼？索引如何找回頁名？ |
| 3 | `retrieve_evidence` 的候選頁面索引與 `topk` | 理解檢索的第二道篩選 | 為何不能從沒有取回的頁面選句？`local_index` 和 `index` 差在哪？ |
| 4 | Training cell 的 `zero_grad → forward → backward → clip → step` | 理解模型何時學習 | 哪一行算 loss？哪一行真的更新參數？ |

有時間再手打 `gold_groups` 和 `evidence_group_hit`：這兩個函式決定「一組證據必須完整命中」及「空集合不能算命中」。讀程式時可以先用 [程式逐段解析](程式逐段解析.md) 對照，再用 [問答筆記](st_question.md) 補觀念。

## 報告數字在哪裡？

| Report 欄位 | 執行後從哪裡拿 |
| --- | --- |
| Q1 文件 Recall@N | 第 5 節 `document_recall`、`document_hits`、`verifiable_dev_samples` 與 `document_top_k` |
| Q2 證據 Recall@N | `evidence_recall`、`evidence_hits`、`verifiable_dev_samples` 與 `evidence_top_k` |
| Q3 Accuracy | `classification_accuracy` 和 `dev_samples` |
| Q4 額外分析 | `document_candidate_coverage` 與 `local_strict_score`；解釋限制，切勿把尚未做的比較寫成實驗結果 |
| Public / private 分數 | Kaggle 該次 submission 的排行榜，不是 dev 分數 |

例如 `0.2500` 才是 25%；它是示意，不是本作業實測成績。若你修改 `DOCUMENT_TOP_K` 或 `EVIDENCE_TOP_K`，報告文字與 `@N` 也要跟著改。前五句是此 notebook 計算本地 strict score 的限制；最終仍以競賽評分器為準。

## 繳交前最後確認

- [ ] Notebook 已完整執行並保留輸出，沒有把小樣本測試的數字誤當完整實驗。
- [ ] Report 的使用者名稱、真實 dev 數字及 public / private score 已依可取得情況填寫；private 在公布後更新。
- [ ] 保留 Report 題目和格式；AI 使用聲明及程式註解與實際情況一致。
- [ ] 使用的資料、每個模型參數上限、禁止外部推理 API 等限制已依助教最新公告核對。資料集與 checkpoint 的公告曾有先後差異，未釐清前僅使用 CFEVER 資料。
- [ ] 依要求選擇最終 Kaggle submissions，並將正確命名的 `.ipynb` 打包成 `.zip`；本資料夾的學習 Markdown 供自己閱讀，作業繳交格式以課程要求為準。
