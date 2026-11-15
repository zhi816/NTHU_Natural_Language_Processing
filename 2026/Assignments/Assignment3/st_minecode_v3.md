<!-- 由 st_minecode_v3.ipynb 自動轉出：markdown 格子原樣保留，程式格子放在 ```python 區塊中。原 notebook 沒有存執行輸出。 -->

# CFEVER 事實查核 v3：三個分類器的 Ensemble

v2（`st_minecode_v2.ipynb`）的 public score 是 **0.856**（第 50 名），dev FEVER score 是 69.00%。v2 的三階段漏斗把 dev 的 2,000 題 SUPPORTS / REFUTES 一路追下去，每一步掉的題數是：

| 階段 | 這一步掉了幾題 |
| :--- | ---: |
| ① 文件檢索：前 5 頁沒找齊證據頁 | 223 |
| ② 證據檢索：頁面對了，前 5 句沒找齊 | 60 |
| ③ **分類：證據已經找齊，標籤卻判錯** | **417** |

另外 NOT ENOUGH INFO 的 1,000 題也錯了 230 題。分類這一步掉的題數比前兩步加起來還多，所以 v3 **文件檢索、證據檢索完全沿用 v2**，只改分類：

```text
claim ─► ① 文件檢索（同 v2）─► ② 證據檢索（同 v2）─► 每題 5 句證據
      ─► ③ 分類：3 個不同的模型各自訓練，再把三類機率平均（ensemble）
      ─► ④ 決策：NEI bias ＋ reranker 門檻（同 v2，搜尋範圍放寬）
```

| | v2 | v3 | 為什麼改 |
| :--- | :--- | :--- | :--- |
| 分類模型 | `chinese-roberta-wwm-ext-large` 1 個 | 再加 `chinese-macbert-large`、`xlm-roberta-large`，共 3 個 | 預訓練方式、斷詞方式都不同，錯的題目不完全一樣，平均起來比較穩 |
| 決策用的分數 | 1 個模型的 logit | 機率平均後取 log；所有模型組合都在 dev 上比過 | 選 dev FEVER score 最高的組合 |
| NEI bias 範圍 | -1.5～1.5 | -3.0～1.0 | v2 選到 -1.5，剛好是範圍的邊界，更小的值可能更好 |
| 門檻分位數 | 每 5% 一格 | 10% 附近每 2.5% 一格 | v2 選到 10%，在附近找細一點 |
| 斷線 | 分類器要整個重訓 | 每訓練完一個模型，就把 dev / test 的預測存成 `pred_*.json`；重跑時自動跳過 | 3 個模型多花 1 個多小時，Colab 斷線不用從頭來 |

輸入格式沒有改。我檢查過 v2 的訓練資料和 test 是不是長得一樣：dev 上證據有找齊的 1,717 題裡，有 1,514 題（88%）的標準證據被 reranker 排在第 1 句，和訓練時「標準證據放最前面」一致。

**作業規範檢查**：5 個模型每一個都 < 1B（bge-m3 568M、bge-reranker-v2-m3 568M、roberta-large 326M、macbert-large 326M、xlm-roberta-large 560M），`count_parameters` 會逐一檢查；助教 11/06 答覆 1B 限制是逐一適用於每個模型。3 個分類器都只做過語言模型預訓練，沒有用其他標註資料集微調過。不呼叫任何外部推理 API；只用 CFEVER。

**執行方式**（詳細步驟在 `index.md` 的 v3 段落）：

- **沿用 v2 的檢索結果（推薦）**：Colab L4 約 1.5～2 小時，只訓練 3 個分類器。步驟寫在第 3 節最後「沿用 v2 檢索結果」那一格。
- **完整重跑**：Run all，Colab L4 約 2.5 小時；Kaggle T4 約 4～5 小時（用 Save & Run All 背景執行）。

## 1. 環境、套件與參數

**[Cell 3｜程式]**

```python
# 生成式 AI 使用說明：這份 notebook 是我在 Claude（Anthropic，Claude Code）協助下改寫的（v1 → v2 → v3 的改進方向也是和 Claude 討論後決定），
# 方法取捨也參考過 ChatGPT、Gemini 的建議；程式由我自己閱讀、執行並核對結果。
# 參考來源：
#   課程 sample code：https://github.com/IKMLab/NTHU_Natural_Language_Processing/tree/main/2026/Assignments/Assignment3
#   CFEVER 資料集：https://github.com/IKMLab/CFEVER-data
#   檢索模型 BAAI/bge-m3：https://huggingface.co/BAAI/bge-m3
#   重排模型 BAAI/bge-reranker-v2-m3：https://huggingface.co/BAAI/bge-reranker-v2-m3
#   reranker 微調方式（1 正例 + 多個 hard negative 的 listwise loss）參考 FlagEmbedding：https://github.com/FlagOpen/FlagEmbedding
#   分類模型（v3 用 3 個做 ensemble）：
#     hfl/chinese-roberta-wwm-ext-large：https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
#     hfl/chinese-macbert-large：https://huggingface.co/hfl/chinese-macbert-large
#     FacebookAI/xlm-roberta-large：https://huggingface.co/FacebookAI/xlm-roberta-large
import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

SEED = 2026


def set_seed(seed):
    """固定亂數種子：抽樣、模型初始化、打亂順序都比較能重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_FP16 = device.type == 'cuda'  # GPU 上用半精度：更快、更省記憶體
print('使用裝置：', device, '| PyTorch', torch.__version__)
if device.type == 'cpu':
    print('⚠️ 沒有偵測到 GPU：完整資料會跑非常久，請在 Kaggle / Colab 開啟 GPU。')
else:
    print('GPU：', torch.cuda.get_device_name(0),
          f'| 記憶體 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')
```

**[Cell 4｜程式]**

```python
# ---- 自動判斷執行環境：Kaggle / Colab / 本機 ----
COLAB_DRIVE_DIR = Path('/content/drive/MyDrive/nlp-hw3')  # Colab：把競賽 ZIP（或 4 個 jsonl）放這裡
REQUIRED_FILES = ['train.jsonl', 'dev.jsonl', 'test.jsonl', 'wiki_pages.jsonl']

try:
    import google.colab  # noqa: F401  只有 Colab 才 import 得到
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
# 注意：新版 Colab 也有一個空的 /kaggle/input 資料夾，所以要先排除 Colab，才能用它判斷 Kaggle
IN_KAGGLE = not IN_COLAB and Path('/kaggle/input').exists()

if IN_KAGGLE:
    DATA_DIR = Path('/kaggle/input')       # 競賽資料要先在右側 Input 加入
    OUTPUT_DIR = Path('/kaggle/working')   # submission.csv 會出現在 Output
elif IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_DIR = COLAB_DRIVE_DIR / 'output_v3'  # v3 的輸出另外放，不會蓋掉 v1 的 output/、v2 的 output_v2/
    if all(any(COLAB_DRIVE_DIR.rglob(name)) for name in REQUIRED_FILES):
        DATA_DIR = COLAB_DRIVE_DIR
    else:
        # Drive 裡只有 ZIP：只解壓需要的 4 個檔到 Colab 本機硬碟（比直接讀 Drive 快）
        DATA_DIR = Path('/content/cfever_data')
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zip_paths = sorted(COLAB_DRIVE_DIR.glob('*.zip'))
        if len(zip_paths) != 1:
            raise FileNotFoundError(f'{COLAB_DRIVE_DIR} 裡要剛好有 1 個競賽 ZIP，目前找到 {len(zip_paths)} 個')
        with zipfile.ZipFile(zip_paths[0]) as archive:
            for name in REQUIRED_FILES:
                target = DATA_DIR / name
                if target.exists():
                    continue
                members = [m for m in archive.infolist() if Path(m.filename).name == name]
                if len(members) != 1:
                    raise FileNotFoundError(f'ZIP 裡應該剛好有 1 份 {name}，目前找到 {len(members)} 份')
                print('解壓：', name)
                partial = DATA_DIR / (name + '.part')  # 先寫 .part，完整寫完才改名，避免斷線留下半個檔案
                with archive.open(members[0]) as source, partial.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                partial.replace(target)
else:
    DATA_DIR = Path('.')
    OUTPUT_DIR = Path('output_v3')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file(filename):
    """在 DATA_DIR（含子資料夾）找檔案；找到 0 份或多份都直接報錯，避免讀錯版本。"""
    direct = DATA_DIR / filename
    if direct.is_file():
        return direct
    matches = sorted(path for path in DATA_DIR.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f'{filename} 找到 {len(matches)} 份，請檢查 DATA_DIR = {DATA_DIR}：{matches}')
    return matches[0]


TRAIN_PATH, DEV_PATH, TEST_PATH, WIKI_PATH = (find_data_file(name) for name in REQUIRED_FILES)
print('環境：', 'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else '本機')
for name, path in zip(REQUIRED_FILES, (TRAIN_PATH, DEV_PATH, TEST_PATH, WIKI_PATH)):
    print(f'  {name:17s} → {path}')
print('輸出資料夾：', OUTPUT_DIR)
```

**[Cell 5｜程式]**

```python
# ---- 資料量（None = 全部）----
TRAIN_SAMPLES = None   # 全部 24,012 筆；只想先確認流程能跑完，可以改成 2000
DEV_SAMPLES = None     # 全部 3,000 筆；bonus / NEI 參數要在完整 dev 上挑，數字才穩
TEST_SAMPLES = None    # 只有試跑才改；不是 None 時不會寫出 submission.csv

# ---- 檢索模型（作業規定：每個模型都要 < 1B 參數）----
RETRIEVER_NAME = 'BAAI/bge-m3'             # 約 568M：頁面與句子的向量
RERANKER_NAME = 'BAAI/bge-reranker-v2-m3'  # 約 568M：句子重排（cross-encoder），會先微調
# 分類模型在下面的 CLASSIFIER_MODELS（v3 用 3 個）

# ---- ① 文件檢索（和 v1、v2 相同）----
MIN_TITLE_LEN = 2         # 太短的標題（1 個字）比對沒有意義
MAX_TITLE_LEN = 30
MAX_TITLE_PAGES = 30      # 每個 claim 最多保留幾個標題命中頁（長標題優先）
DENSE_CANDIDATES = 30     # 每個 claim 再用向量從候選池補幾頁
PAGE_LEAD_CHARS = 200     # 頁面向量只看「頁名 + 開頭 200 字」
DOCUMENT_TOP_K = 5
TITLE_BONUS_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

# ---- ② 證據檢索（和 v2 相同）----
SENTENCE_CANDIDATES = 30  # 向量先留前 30 句，再交給 reranker
EVIDENCE_TOP_K = 5        # FEVER score 只看前 5 句，多給不扣分，所以固定給滿 5 句
EMBED_MAX_LENGTH = 128
EMBED_BATCH_SIZE = 64
RERANK_MAX_LENGTH = 256
RERANK_BATCH_SIZE = 64

# ---- ②-b 微調 reranker（和 v2 相同）----
FINETUNE_RERANKER = True          # False = 直接用現成的 reranker（省約 20 分鐘；報告第 2 題的 (3) 要自己刪掉）
RERANK_TRAIN_CLAIMS = 8000        # 用幾題 train 的 SUPPORTS / REFUTES 來微調
RERANK_NEGATIVES = 7              # 每題 1 個正例 + 7 個 hard negative
RERANK_TRAIN_GROUPS = 4           # 每一步 4 題 × 8 句 = 32 對；GPU 記憶體不夠就改成 2
RERANK_TRAIN_MAX_LENGTH = 192
RERANK_LR = 2e-5
RERANK_TRAIN_NEI_CANDIDATES = 10  # train 的 NEI 題也用 reranker 從向量前 10 句挑 5 句（0 = 不做）

# ---- ③ 分類器（v3：3 個模型各自訓練，再把機率平均）----
CLASSIFIER_MODELS = [
    # name：Hugging Face 上的名字；lr：學習率；freeze_embeddings：凍結詞向量層（省記憶體，和 v2 微調 reranker 的做法一樣）
    {'name': 'hfl/chinese-roberta-wwm-ext-large', 'lr': 1e-5, 'freeze_embeddings': False},  # 326M：v2 用的那個
    {'name': 'hfl/chinese-macbert-large', 'lr': 1e-5, 'freeze_embeddings': False},          # 326M：同大小，預訓練方式不同
    {'name': 'FacebookAI/xlm-roberta-large', 'lr': 8e-6, 'freeze_embeddings': True},        # 560M：多語言，詞表有 25 萬個
]
REUSE_SAVED_PREDICTIONS = True  # 訓練完的模型會把 dev / test 預測存成 pred_*.json；重跑時直接讀檔，不用重訓
EPOCHS = 2
BATCH_SIZE = 8                # large 模型 + 512 token 比較吃記憶體，一次放 8 筆
GRAD_ACCUM_STEPS = 2          # 累積 2 批才更新一次 → 等效 batch size 16
MAX_LENGTH = 512
SENTENCE_CHAR_LIMIT = 120     # 每句證據最多 120 字，避免一句超長的句子把其他句子擠掉
WARMUP_RATIO = 0.1
GRADIENT_CHECKPOINTING = False  # 出現 CUDA out of memory 時改成 True（省記憶體、慢約 30%）

# ---- ④ NEI 決策 ----
NEI_BIAS_GRID = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]  # v2 是 -1.5～1.5，選到邊界的 -1.5，所以往下多試幾格
SCORE_QUANTILES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]  # reranker 門檻 = dev 第 1 名分數的分位數；0 = 不用門檻

# ---- 前兩版實際跑出來的數字（第 7 節的 v1 / v2 / v3 對照表會用到）----
# v1：st_minecode.ipynb 的輸出；v2：st_minecode_v2.ipynb 在 Colab L4 跑出來的 report_filled.md
PREVIOUS_RESULTS = {
    'v1': {'doc_r5': '88.85%', 'ev_r5': '82.70%', 'acc_best': '67.83%', 'fever_best': '62.27%',
           'public': '0.77600', 'rank': 64},
    'v2': {'doc_r5': '88.85%', 'ev_r5': '85.85%', 'acc_best': '73.30%', 'fever_best': '69.00%',
           'public': '0.85600', 'rank': 50, 'fever_raw': 0.69, 'label_correct': 1300, 'nei_correct': 770,
           # v2 第 4 節量的截斷比例；v3 從快取載入時沒有維基句子可以重量，報告就引用這組
           'trunc_v1': '66.50%', 'trunc_v2': '100.00%', 'median_len': 287, 'over256': '66.45%',
           'trunc_v1_raw': 0.665, 'n_trunc_sample': 2000},
}

LABELS = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']  # 分類器輸出 0 / 1 / 2 的意思
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
NEI = 'NOT ENOUGH INFO'
NEI_ID = LABEL_TO_ID[NEI]
SUBMIT_LABEL = {'SUPPORTS': 'supports', 'REFUTES': 'refutes', NEI: NEI}  # 與 sample code 的提交格式相同
```

### 讀取資料

JSONL 一行就是一筆 JSON。`all_train_df` 保留全部 train（用來建立候選頁），`train_df` 是這次拿來訓練分類器的資料。
test 只保留 `id` 和 `claim`，整份 notebook 都不會碰 test 的答案。

**[Cell 7｜程式]**

```python
def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as file:
        return pd.read_json(file, lines=True)


all_train_df = read_jsonl(TRAIN_PATH)
dev_df = read_jsonl(DEV_PATH)
test_df = read_jsonl(TEST_PATH)[['id', 'claim']]
for df in (all_train_df, dev_df):
    df['label'] = df['label'].astype(str).str.upper()
for df in (all_train_df, dev_df, test_df):
    df['claim'] = df['claim'].astype(str)

train_df = all_train_df
if TRAIN_SAMPLES is not None:
    train_df = all_train_df.sample(n=min(TRAIN_SAMPLES, len(all_train_df)), random_state=SEED)
if DEV_SAMPLES is not None:
    dev_df = dev_df.sample(n=min(DEV_SAMPLES, len(dev_df)), random_state=SEED)
if TEST_SAMPLES is not None:
    test_df = test_df.head(TEST_SAMPLES)
# 重設索引：之後第 i 筆資料就對應第 i 個檢索結果
train_df = train_df.reset_index(drop=True)
dev_df = dev_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print('train / dev / test 筆數：', len(train_df), len(dev_df), len(test_df))
print('train 標籤分布：', dict(Counter(train_df['label'])))
print('dev 標籤分布：  ', dict(Counter(dev_df['label'])))
train_df[['id', 'claim', 'label', 'evidence']].head()
```

## 2. Document Retrieval：找出相關的維基頁面

標準證據的每一條是 `[annotation_id, evidence_id, 頁名, 句號]`；`evidence` 是好幾「組」，**找齊任一組**就算對。
下面先寫好幾個整份 notebook 都會用到的小工具：整理標準證據、正規化標題、計算 Recall、印 Markdown 表格。

**[Cell 9｜程式]**

```python
def gold_groups(evidence):
    """標準證據 → [ {(頁名, 句號), ...}, ... ]，一個 set 是一組要「一起找到」的證據。
    NEI 的證據長得像 [[[標註id, None, None, None]]]，會被略過；空組也略過（空集合.issubset 永遠是 True）。"""
    groups = []
    if not isinstance(evidence, list):
        return groups
    for group in evidence:
        pairs = set()
        for item in group if isinstance(group, list) else []:
            valid = (isinstance(item, list) and len(item) >= 4 and isinstance(item[2], str)
                     and isinstance(item[3], (int, float)) and not pd.isna(item[3]))
            if not valid:  # 一組裡只要有一條壞掉，整組都不算
                pairs = set()
                break
            pairs.add((item[2], int(item[3])))
        if pairs:
            groups.append(pairs)
    return groups


def is_verifiable(label):
    return label in ('SUPPORTS', 'REFUTES')


def normalize_text(text):
    """比對用的正規化：英文小寫、各種間隔號統一成「·」、去掉所有空白。"""
    text = re.sub(r'[·‧・•]', '·', text.lower())
    return re.sub(r'\s+', '', text)


def display_title(page_id):
    """頁名給人看 / 給模型讀的樣子（處理底線與 -LRB- -RRB-）。"""
    return page_id.replace('-LRB-', '(').replace('-RRB-', ')').replace('_', ' ')


def title_key(page_id):
    """頁名 → 會出現在 claim 裡的樣子：去掉結尾的消歧義括號，例如「水星 (行星)」→「水星」。"""
    title = re.sub(r'\s*[（(][^（）()]*[）)]\s*$', '', display_title(page_id))
    return normalize_text(title)


def doc_hit(groups, pages):
    """前 N 頁是否涵蓋「任一組」標準證據需要的「所有」頁面。"""
    pages = set(pages)
    return any({title for title, _ in group} <= pages for group in groups)


def evidence_hit(groups, evidence):
    """前 N 句（[頁名, 句號]）是否完整涵蓋任一組標準證據；頁名、句號都要對。"""
    found = {(title, int(sentence_id)) for title, sentence_id in evidence}
    return any(group <= found for group in groups)


def recall_at(df, predictions, hit_fn, k=None):
    """只算 SUPPORTS / REFUTES。回傳 (recall, 命中題數, 可驗證題數)。"""
    hits = total = 0
    for label, evidence, predicted in zip(df['label'], df['evidence'], predictions):
        if not is_verifiable(label):
            continue
        total += 1
        hits += hit_fn(gold_groups(evidence), predicted if k is None else predicted[:k])
    return hits / max(total, 1), hits, total


def pct(value):
    return f'{value * 100:.2f}%'


def md_table(df):
    """DataFrame → Markdown 表格字串（可以直接貼進 Report）。"""
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |',
             '| ' + ' | '.join(':---:' for _ in df.columns) + ' |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


report_values = {}  # 一路收集報告要用的數字，第 7 節會自動填進報告


def resolve_model(name):
    """Kaggle 若已把模型加進 Input（離線用），就用 Input 裡的資料夾；否則從 Hugging Face 下載（要開 Internet）。"""
    short_name = name.split('/')[-1].lower()
    if IN_KAGGLE:
        for config_path in sorted(Path('/kaggle/input').rglob('config.json')):
            if short_name in str(config_path.parent).lower():
                return str(config_path.parent)
    return name


def count_parameters(model):
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count < 1_000_000_000, f'模型參數 {count:,} 超過 1B，違反作業規定'
    return count
```

### 第 1 次讀維基：只讀頁名，建立「標題 → 頁面」索引

維基有一百多萬頁、約 1 GB，全部丟進 BERT 太花 GPU 時間。可是 claim 裡的主角名稱常常就是頁名，
所以我先只讀頁名，做成一個 dict（Python 的 dict 查詢是 O(1)，一百多萬個 key 也很快）。

**[Cell 11｜程式]**

```python
start_time = time.time()
title_index = defaultdict(list)  # 正規化標題 → [原始頁名, ...]（同一標題可能有好幾個消歧義頁）
wiki_page_count = 0
with WIKI_PATH.open(encoding='utf-8') as file:
    for line in file:
        if not line.strip():
            continue
        page_id = json.loads(line).get('id') or ''
        wiki_page_count += 1
        if not page_id or '消歧義' in page_id or '消歧义' in page_id:
            continue
        key = title_key(page_id)
        if MIN_TITLE_LEN <= len(key) <= MAX_TITLE_LEN:
            title_index[key].append(page_id)
title_index = dict(title_index)  # 之後只查不寫，轉回一般 dict 比較保險
print(f'維基共 {wiki_page_count:,} 頁；可比對的標題 {len(title_index):,} 個；耗時 {time.time() - start_time:.0f} 秒')
report_values['n_wiki_pages'] = f'{wiki_page_count:,}'
```

### 標題比對：claim 的每個子字串都去查表

例如 claim = 「周杰倫發行了專輯」，就去查「周杰」「周杰倫」「周杰倫發」…「專輯」，查得到的都是候選頁。
長的標題比較具體（「周杰倫」比「專輯」有用），所以排前面，每題最多留 `MAX_TITLE_PAGES` 頁。

**[Cell 13｜程式]**

```python
def match_titles(claim):
    text = normalize_text(claim)
    found = {}  # 頁名 → 命中的標題長度
    for start in range(len(text)):
        for end in range(start + MIN_TITLE_LEN, min(len(text), start + MAX_TITLE_LEN) + 1):
            for page_id in title_index.get(text[start:end], []):
                found[page_id] = max(found.get(page_id, 0), end - start)
    # 長標題優先；同長度時，沒有消歧義括號的頁排前面
    ranked = sorted(found, key=lambda page_id: (-found[page_id], '(' in display_title(page_id), page_id))
    return ranked[:MAX_TITLE_PAGES]


start_time = time.time()
train_title_matches = [match_titles(claim) for claim in train_df['claim']]
dev_title_matches = [match_titles(claim) for claim in dev_df['claim']]
test_title_matches = [match_titles(claim) for claim in test_df['claim']]
print(f'標題比對完成，耗時 {time.time() - start_time:.0f} 秒')

title_coverage = recall_at(dev_df, dev_title_matches, doc_hit)[0]
no_match = sum(not matches for matches in dev_title_matches)
print('範例 claim：', dev_df.loc[0, 'claim'])
print('標題命中：', dev_title_matches[0][:10])
print(f'dev 平均每題命中 {np.mean([len(m) for m in dev_title_matches]):.1f} 頁；完全沒命中的有 {no_match} 題')
print(f'只靠標題比對（最多 {MAX_TITLE_PAGES} 頁）就涵蓋標準證據頁的比例：{pct(title_coverage)}')
report_values.update(
    avg_title_matches=f'{np.mean([len(m) for m in dev_title_matches]):.1f}',
    no_title_match=f'{no_match} 題（{pct(no_match / max(len(dev_df), 1))}）',
    title_coverage=pct(title_coverage),
)
```

### 第 2 次讀維基：只載入「候選池」的頁面內容

候選池 = train 標註過的頁（sample code 原本就用的）∪ 所有 claim 的標題命中頁。
只有這些頁需要內容：開頭文字拿去算頁面向量，`lines` 拆成句子留給第 3 節。
`lines` 每一行是「句號\t句子\t（可能還有超連結）」，**一定要保留原始句號**，不能自己重新編號。

**[Cell 15｜程式]**

```python
def parse_lines(lines):
    sentences = []
    for line in lines.split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2 and parts[0].strip().isdecimal() and parts[1].strip():
            sentences.append((int(parts[0]), parts[1].strip()))
    return sentences


train_gold_pages = {title for evidence in all_train_df['evidence']
                    for group in gold_groups(evidence) for title, _ in group}
matched_pages = {page_id for matches in (train_title_matches, dev_title_matches, test_title_matches)
                 for claim_matches in matches for page_id in claim_matches}
wanted_pages = train_gold_pages | matched_pages

start_time = time.time()
page_lead = {}       # 頁名 → 開頭文字（給頁面向量用）
page_sentences = {}  # 頁名 → [(原始句號, 句子), ...]
with WIKI_PATH.open(encoding='utf-8') as file:
    for line in file:
        if not line.strip():
            continue
        page = json.loads(line)
        if page.get('id') not in wanted_pages:
            continue
        sentences = parse_lines(page.get('lines') or '')
        text = page.get('text') or ' '.join(sentence for _, sentence in sentences)
        page_lead[page['id']] = text[:PAGE_LEAD_CHARS]
        page_sentences[page['id']] = sentences
print(f'train 標註頁 {len(train_gold_pages):,} + 標題命中頁 {len(matched_pages):,} → 候選池 {len(page_sentences):,} 頁'
      f'（耗時 {time.time() - start_time:.0f} 秒）')
print('train 標註頁在維基裡找不到的數量：', len(train_gold_pages - page_sentences.keys()))
report_values.update(pool_size=f'{len(page_sentences):,}', n_train_gold_pages=f'{len(train_gold_pages):,}')
```

### 用 bge-m3 把文字變成向量

sample code 用 bert-base-chinese 做 mean pooling，但原生 BERT 沒有為「找相似文字」訓練過，
向量都擠在差不多的方向（anisotropy），cosine 分不太出好壞。`BAAI/bge-m3` 是專門用對比學習訓練的檢索模型，
它訓練時就是拿 **[CLS] 位置**的向量，所以這裡照它的用法取 [CLS]（`pooling='cls'`）；`'mean'` 的寫法也保留下來對照。
向量做 L2 normalization 後長度是 1，**內積就等於 cosine similarity**。

兩個加速技巧：(1) GPU 上用 fp16；(2) 先把文字依長度排序再分批，同一批長度差不多，padding 浪費很少。

**[Cell 17｜程式]**

```python
retriever_path = resolve_model(RETRIEVER_NAME)
retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_path)
retriever = AutoModel.from_pretrained(retriever_path).to(device).eval()
if USE_FP16:
    retriever.half()
retriever_params = count_parameters(retriever)
print(f'檢索模型 {RETRIEVER_NAME}：{retriever_params:,} 參數')
EMBED_DTYPE = torch.float16 if USE_FP16 else torch.float32  # 向量存在 CPU 上，fp16 省一半記憶體


@torch.no_grad()
def encode_texts(texts, pooling='cls', name=''):
    """文字清單 → [文字數, 向量維度] 的 tensor（已 L2 正規化，存在 CPU）。"""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # 依長度排序，padding 最少
    vectors = torch.empty(len(texts), retriever.config.hidden_size, dtype=EMBED_DTYPE)
    for batch_no, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE)):
        index = order[start:start + EMBED_BATCH_SIZE]
        inputs = retriever_tokenizer([texts[i] for i in index], padding=True, truncation=True,
                                     max_length=EMBED_MAX_LENGTH, return_tensors='pt').to(device)
        hidden = retriever(**inputs).last_hidden_state  # [batch, token 數, 維度]
        if pooling == 'cls':
            pooled = hidden[:, 0]                       # 第 0 個 token = [CLS]（XLM-R 叫 <s>）
        else:                                           # mean pooling：只平均真正的 token，不算 padding
            mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled.float(), p=2, dim=1)  # 長度變 1 → 內積 = cosine
        vectors[index] = pooled.to(EMBED_DTYPE).cpu()       # 放回「原本的順序」
        if (batch_no + 1) % 500 == 0:
            print(f'  {name} 已編碼 {min(start + EMBED_BATCH_SIZE, len(texts)):,} / {len(texts):,}')
    return vectors
```

**[Cell 18｜程式]**

```python
pool_pages = sorted(page_sentences)                         # 頁面向量的第 i 列 = pool_pages[i]
page_row = {page_id: row for row, page_id in enumerate(pool_pages)}
page_texts = [f'{display_title(page_id)}：{page_lead[page_id]}' for page_id in pool_pages]

start_time = time.time()
page_embeddings = encode_texts(page_texts, name='頁面')
train_claim_emb = encode_texts(train_df['claim'].tolist(), name='train claim')
dev_claim_emb = encode_texts(dev_df['claim'].tolist(), name='dev claim')
test_claim_emb = encode_texts(test_df['claim'].tolist(), name='test claim')
print(f'頁面向量 {tuple(page_embeddings.shape)}，耗時 {time.time() - start_time:.0f} 秒')
```

### 混合分數：cosine + 標題加分

每個 claim 的候選頁 = 向量最像的前 `DENSE_CANDIDATES` 頁 ∪ 標題命中頁。
最終分數 = cosine similarity + bonus × [這頁是否被標題命中]，取前 `DOCUMENT_TOP_K` 頁。
bonus = 0 等於只用向量；bonus 很大等於「標題命中的頁一定排前面」。我在 dev 上把幾個 bonus 都試一次，選 Recall 最高的。

**[Cell 20｜程式]**

```python
@torch.no_grad()
def page_candidates(claim_embeddings, title_matches):
    """回傳每個 claim 的候選頁：{頁面列號: (cosine, 是否標題命中)}。"""
    page_matrix = page_embeddings.to(device)
    results = []
    for start in range(0, len(claim_embeddings), 256):
        chunk = claim_embeddings[start:start + 256].to(device)
        similarities = chunk @ page_matrix.T  # [這批 claim 數, 候選池頁數]
        top = similarities.topk(k=min(DENSE_CANDIDATES, similarities.shape[1]), dim=1)
        top_values, top_rows = top.values.float().cpu().tolist(), top.indices.cpu().tolist()
        for offset in range(len(chunk)):
            claim_no = start + offset
            candidates = {row: (score, False) for score, row in zip(top_values[offset], top_rows[offset])}
            matched_rows = [page_row[page_id] for page_id in title_matches[claim_no] if page_id in page_row]
            if matched_rows:
                scores = page_embeddings[matched_rows].float() @ claim_embeddings[claim_no].float()
                for row, score in zip(matched_rows, scores.tolist()):
                    candidates[row] = (score, True)
            results.append(candidates)
    del page_matrix
    return results


def rank_pages(candidates, bonus, k=DOCUMENT_TOP_K):
    ranked = sorted(candidates.items(), key=lambda item: item[1][0] + bonus * item[1][1], reverse=True)
    return [pool_pages[row] for row, _ in ranked[:k]]


train_page_cands = page_candidates(train_claim_emb, train_title_matches)
dev_page_cands = page_candidates(dev_claim_emb, dev_title_matches)
test_page_cands = page_candidates(test_claim_emb, test_title_matches)

# 在 dev 上挑 bonus
bonus_rows = []
for bonus in TITLE_BONUS_GRID:
    recall, hits, total = recall_at(dev_df, [rank_pages(c, bonus) for c in dev_page_cands], doc_hit)
    bonus_rows.append({'title bonus': bonus, f'Doc Recall@{DOCUMENT_TOP_K}': pct(recall), '命中題數': f'{hits} / {total}',
                       '_recall': recall})
best_bonus = max(bonus_rows, key=lambda row: row['_recall'])['title bonus']  # 同分時取較小的 bonus
bonus_table = pd.DataFrame(bonus_rows).drop(columns='_recall')
print(md_table(bonus_table))
print('選用的 title bonus =', best_bonus)

# 三種做法、不同 N 的 Recall@N
doc_methods = {
    '只用標題比對（長標題優先）': dev_title_matches,
    '只用 bge-m3 向量（bonus = 0）': [rank_pages(c, 0.0, k=10) for c in dev_page_cands],
    f'混合（bonus = {best_bonus}）': [rank_pages(c, best_bonus, k=10) for c in dev_page_cands],
}
doc_rows = []
for method, predictions in doc_methods.items():
    row = {'方法': method}
    for k in (1, 3, 5, 10):
        row[f'R@{k}'] = pct(recall_at(dev_df, predictions, doc_hit, k=k)[0])
    doc_rows.append(row)
doc_table = pd.DataFrame(doc_rows)
candidate_coverage = recall_at(dev_df, [[pool_pages[row] for row in c] for c in dev_page_cands], doc_hit)[0]
print(md_table(doc_table))
print('候選頁（向量 + 標題）涵蓋率：', pct(candidate_coverage))

train_docs = [rank_pages(c, best_bonus) for c in train_page_cands]
dev_docs = [rank_pages(c, best_bonus) for c in dev_page_cands]
test_docs = [rank_pages(c, best_bonus) for c in test_page_cands]
doc_recall, doc_hits, verifiable_count = recall_at(dev_df, dev_docs, doc_hit)
print(f'Document Retrieval Recall@{DOCUMENT_TOP_K}: {doc_recall:.4f}（{doc_hits} / {verifiable_count}）')
print('範例 claim：', test_df.loc[0, 'claim'], '\n取回頁面：', test_docs[0])

title_only_r5 = recall_at(dev_df, doc_methods['只用標題比對（長標題優先）'], doc_hit, k=DOCUMENT_TOP_K)[0]
dense_only_r5 = recall_at(dev_df, doc_methods['只用 bge-m3 向量（bonus = 0）'], doc_hit, k=DOCUMENT_TOP_K)[0]
max_bonus_r5 = next(row['_recall'] for row in bonus_rows if row['title bonus'] == max(TITLE_BONUS_GRID))
report_values.update(
    retriever_params=f'{retriever_params / 1e6:.0f}M', best_bonus=best_bonus, n_dev=len(dev_df),
    n_verifiable=verifiable_count, doc_r5=pct(doc_recall), doc_hits=doc_hits, doc_table=md_table(doc_table),
    bonus_table=md_table(bonus_table), candidate_coverage=pct(candidate_coverage),
    doc_title_only=pct(title_only_r5), doc_dense_only=pct(dense_only_r5), doc_max_bonus=pct(max_bonus_r5),
    doc_recall_raw=doc_recall, doc_title_only_raw=title_only_r5, doc_dense_only_raw=dense_only_r5,
    doc_max_bonus_raw=max_bonus_r5,
)
```

## 3. Evidence Retrieval：從取回的頁面挑句子

規定是**只能從第 1 階段取回的頁面選句**。和 v1 一樣，這些頁的每一句都是候選，句子前面加上頁名（很多句子用「他」「該片」代稱主角）。
v2 的改變在後半段：初篩留 30 句，而且 reranker 會先用 train 的標準證據微調。

**[Cell 22｜程式]**

```python
needed_pages = sorted({page_id for docs in (train_docs, dev_docs, test_docs) for pages in docs for page_id in pages})
sentence_keys, sentence_texts = [], []  # 第 i 句的 (頁名, 原始句號) 和要給模型讀的文字
page_sentence_rows = {}                 # 頁名 → 這頁每一句在上面兩個 list 的位置
for page_id in needed_pages:
    rows = []
    for sentence_id, sentence in page_sentences.get(page_id, []):
        rows.append(len(sentence_keys))
        sentence_keys.append((page_id, sentence_id))
        sentence_texts.append(f'{display_title(page_id)}：{sentence}')
    page_sentence_rows[page_id] = rows
print(f'需要的頁面 {len(needed_pages):,} 頁，候選句 {len(sentence_texts):,} 句')

start_time = time.time()
sentence_embeddings = encode_texts(sentence_texts, name='句子')
print(f'句子向量 {tuple(sentence_embeddings.shape)}，耗時 {time.time() - start_time:.0f} 秒')
report_values['n_sentences'] = f'{len(sentence_texts):,}'

# 所有向量都算完了：釋放檢索模型的 GPU 記憶體，留給後面的模型
del retriever
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

page_sentence_dict = {}


def get_sentence(page_id, sentence_id):
    """(頁名, 句號) → 句子內容；找不到回傳 None。"""
    if page_id not in page_sentence_dict:
        page_sentence_dict[page_id] = dict(page_sentences.get(page_id, []))
    return page_sentence_dict[page_id].get(sentence_id)


def sentence_text(key):
    """(頁名, 句號) → 「頁名：句子」，和 sentence_texts 的格式一樣（reranker 讀的就是這個）。"""
    page_id, sentence_id = key
    return f'{display_title(page_id)}：{get_sentence(page_id, sentence_id)}'
```

### 第一關：向量初篩（30 句）

每個 claim 只看「自己那 5 頁」的句子。`scores.topk` 回傳的是**候選清單內的位置**（local），要用 `rows[local]` 換回全體句子清單的位置。
v2 連 train 也取前 30 句：等一下微調 reranker 要從裡面挑「很像、但不是標準證據」的難負例。

**[Cell 24｜程式]**

```python
def rank_sentences_dense(claim_embeddings, docs, top_n):
    """回傳每個 claim 最像的 top_n 句（句子的全域列號）。"""
    results = []
    for claim_no, pages in enumerate(docs):
        rows = [row for page_id in pages for row in page_sentence_rows.get(page_id, [])]
        if not rows:
            results.append([])
            continue
        scores = sentence_embeddings[rows].float() @ claim_embeddings[claim_no].float()
        best_local = scores.topk(k=min(top_n, len(rows))).indices.tolist()
        results.append([rows[local] for local in best_local])  # local（候選內位置）→ 全域列號
    return results


def rows_to_keys(rows_list, k):
    return [[sentence_keys[row] for row in rows[:k]] for rows in rows_list]


train_dense_rows = rank_sentences_dense(train_claim_emb, train_docs, SENTENCE_CANDIDATES)
dev_dense_rows = rank_sentences_dense(dev_claim_emb, dev_docs, SENTENCE_CANDIDATES)
test_dense_rows = rank_sentences_dense(test_claim_emb, test_docs, SENTENCE_CANDIDATES)
dev_evidence_dense = rows_to_keys(dev_dense_rows, EVIDENCE_TOP_K)
print('向量初篩的前 5 句：', dev_evidence_dense[0])

# 向量用完了：句子向量可能有好幾 GB，先釋放 RAM 給後面的 large 模型
del sentence_embeddings, page_embeddings, train_claim_emb, dev_claim_emb, test_claim_emb
gc.collect()
```

### 第二關：cross-encoder 重排（先看現成模型的表現）

和 v1 一樣用 `bge-reranker-v2-m3`，差別是 v2 用 **fp32 載入**（等一下要訓練它，fp16 權重直接訓練容易不穩），推論時再用 `torch.autocast` 轉半精度加速。
`rerank` 除了回傳每題前 5 句，也回傳**第 1 名的分數**：連最像的句子分數都很低，代表這題可能根本找不到證據，第 5 節會拿來幫忙判斷 NEI。

**[Cell 26｜程式]**

```python
reranker_path = resolve_model(RERANKER_NAME)
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path)


def load_reranker():
    model = AutoModelForSequenceClassification.from_pretrained(reranker_path).to(device)
    return model.eval()


@torch.no_grad()
def rerank(model, claims, candidate_rows, top_k):
    """回傳 (每題重排後的前 top_k 句列號, 每題第 1 名的分數)。"""
    model.eval()
    pairs = [(claim_no, row) for claim_no, rows in enumerate(candidate_rows) for row in rows]
    scores = torch.empty(len(pairs))
    order = sorted(range(len(pairs)), key=lambda i: len(claims[pairs[i][0]]) + len(sentence_texts[pairs[i][1]]))
    for batch_no, start in enumerate(range(0, len(pairs), RERANK_BATCH_SIZE)):
        index = order[start:start + RERANK_BATCH_SIZE]
        inputs = reranker_tokenizer([claims[pairs[i][0]] for i in index],
                                    [sentence_texts[pairs[i][1]] for i in index],
                                    padding=True, truncation='only_second',
                                    max_length=RERANK_MAX_LENGTH, return_tensors='pt').to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
            scores[index] = model(**inputs).logits.view(-1).float().cpu()  # 一對一個相關分數
        if (batch_no + 1) % 500 == 0:
            print(f'  已重排 {min(start + RERANK_BATCH_SIZE, len(pairs)):,} / {len(pairs):,} 對')
    by_claim = defaultdict(list)
    for (claim_no, row), score in zip(pairs, scores.tolist()):
        by_claim[claim_no].append((score, row))
    top_rows, top_scores = [], []
    for claim_no in range(len(candidate_rows)):
        ranked = sorted(by_claim[claim_no], reverse=True)
        top_rows.append([row for _, row in ranked[:top_k]])
        top_scores.append(ranked[0][0] if ranked else -1e9)  # 沒有任何候選句 → 給一個極低分
    return top_rows, top_scores


reranker = load_reranker()
reranker_params = count_parameters(reranker)
print(f'重排模型 {RERANKER_NAME}：{reranker_params:,} 參數')
dev_claims, test_claims = dev_df['claim'].tolist(), test_df['claim'].tolist()

start_time = time.time()
dev_rows_zs, dev_scores_zs = rerank(reranker, dev_claims, dev_dense_rows, EVIDENCE_TOP_K)
zs_recall = recall_at(dev_df, rows_to_keys(dev_rows_zs, EVIDENCE_TOP_K), evidence_hit)[0]
print(f'現成 reranker 的 dev Evidence Recall@{EVIDENCE_TOP_K} = {pct(zs_recall)}（耗時 {time.time() - start_time:.0f} 秒）')
```

### 微調 reranker：讓它學 CFEVER 的標註習慣

每一筆訓練資料是一「組」8 句：第 0 句是標準證據（正例），後面 7 句是 bge-m3 初篩排很前面、但不是標準證據的句子
（**hard negative**：看起來很像，其實不是答案）。
reranker 對 8 句各打一個分數，把這 8 個分數當成「8 選 1」的 logit 做 cross entropy，目標永遠是第 0 個，
意思就是「標準證據的分數要比其他 7 句都高」。這叫 listwise loss，bge-reranker 原本也是用這種方式訓練的。

只用 train 的 SUPPORTS / REFUTES（最多 `RERANK_TRAIN_CLAIMS` 題），**dev 完全沒有拿來訓練**。

**[Cell 28｜程式]**

```python
rng = random.Random(SEED)
rerank_examples = []  # (claim, [正例句, 負例句 × RERANK_NEGATIVES])
for claim, label, evidence, rows in zip(train_df['claim'], train_df['label'], train_df['evidence'], train_dense_rows):
    if not is_verifiable(label):
        continue
    gold = sorted({key for group in gold_groups(evidence) for key in group if get_sentence(*key)})
    negatives = [row for row in rows if sentence_keys[row] not in gold][:RERANK_NEGATIVES * 2]  # 只從最像的前 14 句裡挑
    if not gold or len(negatives) < RERANK_NEGATIVES:
        continue
    positive = rng.choice(gold)
    picked = rng.sample(negatives, RERANK_NEGATIVES)
    rerank_examples.append((claim, [sentence_text(positive)] + [sentence_texts[row] for row in picked]))
rng.shuffle(rerank_examples)
rerank_examples = rerank_examples[:RERANK_TRAIN_CLAIMS]
print(f'微調資料 {len(rerank_examples):,} 組（每組 1 個正例 + {RERANK_NEGATIVES} 個 hard negative）')
if rerank_examples:
    claim, texts = rerank_examples[0]
    print('範例 claim：', claim, '\n  正例：', texts[0], '\n  負例：', texts[1])
```

兩個讓 T4（16 GB）放得下的做法：
- **凍結詞向量層**：XLM-R 的詞表有 25 萬個詞，光是詞向量就佔了約 2.5 億參數。凍結後不用存它們的梯度和 Adam 狀態，記憶體省一大截；
  詞義本身也不需要為了這個任務改變，要學的是「怎麼比對 claim 和句子」，那是後面 transformer 層的工作。
- **fp16 混合精度**：和第 4 節的分類器一樣，`autocast` + `GradScaler`。

微調完會在 dev 上和現成模型比較：**如果反而變差，就自動改回現成的 reranker**，所以開這個功能不會比 v1 更糟。

**[Cell 30｜程式]**

```python
def finetune_reranker(model, examples):
    for parameter in model.base_model.embeddings.parameters():
        parameter.requires_grad = False  # 凍結詞向量層
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f'可訓練參數 {sum(p.numel() for p in trainable):,} / 全部 {sum(p.numel() for p in model.parameters()):,}')
    optimizer = torch.optim.AdamW(trainable, lr=RERANK_LR, weight_decay=0.01)
    total_steps = math.ceil(len(examples) / RERANK_TRAIN_GROUPS)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_FP16)
    group_size = 1 + RERANK_NEGATIVES
    model.train()
    total_loss, start_time = 0.0, time.time()
    for step, start in enumerate(range(0, len(examples), RERANK_TRAIN_GROUPS), start=1):
        batch = examples[start:start + RERANK_TRAIN_GROUPS]
        inputs = reranker_tokenizer([claim for claim, texts in batch for _ in texts],   # claim 重複 8 次
                                    [text for _, texts in batch for text in texts],      # 每組的 8 句攤平
                                    padding=True, truncation='only_second',
                                    max_length=RERANK_TRAIN_MAX_LENGTH, return_tensors='pt').to(device)
        target = torch.zeros(len(batch), dtype=torch.long, device=device)  # 正例永遠在第 0 個
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
            scores = model(**inputs).logits.view(len(batch), group_size)  # [題數 × 8, 1] → [題數, 8]
        loss = F.cross_entropy(scores.float(), target)                     # 8 選 1，答案是第 0 個
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        if step % 200 == 0 or step == total_steps:
            print(f'  reranker step {step}/{total_steps}  平均 loss = {total_loss / step:.4f}'
                  f'  （{time.time() - start_time:.0f} 秒）')
    model.eval()
    return total_loss / max(total_steps, 1)


reranker_choice = '現成的'
dev_final_rows, dev_top_scores = dev_rows_zs, dev_scores_zs
ft_recall, rerank_minutes = None, 0.0
if FINETUNE_RERANKER and rerank_examples:
    start_time = time.time()
    finetune_reranker(reranker, rerank_examples)
    rerank_minutes = (time.time() - start_time) / 60
    dev_rows_ft, dev_scores_ft = rerank(reranker, dev_claims, dev_dense_rows, EVIDENCE_TOP_K)
    ft_recall = recall_at(dev_df, rows_to_keys(dev_rows_ft, EVIDENCE_TOP_K), evidence_hit)[0]
    print(f'微調花了 {rerank_minutes:.0f} 分鐘；dev Evidence Recall@{EVIDENCE_TOP_K}：'
          f'現成 {pct(zs_recall)} → 微調後 {pct(ft_recall)}')
    if ft_recall >= zs_recall:
        reranker_choice = '微調後的'
        dev_final_rows, dev_top_scores = dev_rows_ft, dev_scores_ft
    else:
        print('微調後反而變差，改回現成的 reranker')
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reranker = load_reranker()
print('之後使用', reranker_choice, 'reranker')
```

### 用選好的 reranker 處理 test，以及 train 的 NEI 題

- test：和 dev 一樣，從向量前 30 句挑 5 句。
- train 的 NEI 題：從向量前 10 句挑 5 句。這樣分類器訓練時看到的 NEI 證據，會和 test 時一樣「看起來很相關、但其實不夠判斷」。
  （train 的 S/R 題會在下面補進標準證據，所以不用重排。）

**[Cell 32｜程式]**

```python
start_time = time.time()
test_final_rows, test_top_scores = rerank(reranker, test_claims, test_dense_rows, EVIDENCE_TOP_K)
train_context_rows = [rows[:EVIDENCE_TOP_K] for rows in train_dense_rows]
if RERANK_TRAIN_NEI_CANDIDATES:
    nei_index = [i for i, label in enumerate(train_df['label']) if label == NEI]
    nei_rows, _ = rerank(reranker, [train_df['claim'][i] for i in nei_index],
                         [train_dense_rows[i][:RERANK_TRAIN_NEI_CANDIDATES] for i in nei_index], EVIDENCE_TOP_K)
    for i, rows in zip(nei_index, nei_rows):
        train_context_rows[i] = rows
print(f'test 與 train NEI 重排完成，耗時 {time.time() - start_time:.0f} 秒')
del reranker
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

dev_evidence = rows_to_keys(dev_final_rows, EVIDENCE_TOP_K)
test_evidence = rows_to_keys(test_final_rows, EVIDENCE_TOP_K)
print('證據座標：', test_evidence[0])
```

### Evidence Retrieval Recall@5

一樣只算 SUPPORTS / REFUTES，前 5 句要**完整**涵蓋某一組標準證據（頁名、句號都要對）。
另外算「文件已經找對的題目」裡的證據命中率，才分得出是哪一階段出問題。

**[Cell 34｜程式]**

```python
evidence_recall, evidence_hits, _ = recall_at(dev_df, dev_evidence, evidence_hit)
dense_recall = recall_at(dev_df, dev_evidence_dense, evidence_hit)[0]
candidate_recall_20 = recall_at(dev_df, rows_to_keys(dev_dense_rows, 20), evidence_hit)[0]
candidate_recall = recall_at(dev_df, rows_to_keys(dev_dense_rows, SENTENCE_CANDIDATES), evidence_hit)[0]

doc_ok = evidence_ok = 0
for label, evidence, docs, keys in zip(dev_df['label'], dev_df['evidence'], dev_docs, dev_evidence):
    if is_verifiable(label) and doc_hit(gold_groups(evidence), docs):
        doc_ok += 1
        evidence_ok += evidence_hit(gold_groups(evidence), keys)
evidence_given_doc = evidence_ok / max(doc_ok, 1)

column = f'Evidence Recall@{EVIDENCE_TOP_K}'
evidence_rows = [
    {'方法': 'bge-m3 向量取前 5 句', column: pct(dense_recall)},
    {'方法': 'bge-m3 向量前 20 句（v1 reranker 能到的上限）', column: pct(candidate_recall_20)},
    {'方法': f'bge-m3 向量前 {SENTENCE_CANDIDATES} 句（v2 reranker 能到的上限）', column: pct(candidate_recall)},
    {'方法': f'＋ 現成 bge-reranker 從 {SENTENCE_CANDIDATES} 句重排取前 5', column: pct(zs_recall)},
]
if ft_recall is not None:
    evidence_rows.append({'方法': f'＋ 微調後 bge-reranker 從 {SENTENCE_CANDIDATES} 句重排取前 5', column: pct(ft_recall)})
evidence_table = pd.DataFrame(evidence_rows)
print(md_table(evidence_table))
print(f'Evidence Retrieval Recall@{EVIDENCE_TOP_K}: {evidence_recall:.4f}（{evidence_hits} / {verifiable_count}，{reranker_choice} reranker）')
print(f'文件已找對的 {doc_ok} 題中，證據也找齊的比例：{pct(evidence_given_doc)}')
report_values.update(
    reranker_params=f'{reranker_params / 1e6:.0f}M', reranker_choice=reranker_choice,
    ev_r5=pct(evidence_recall), ev_hits=evidence_hits, ev_dense_r5=pct(dense_recall),
    ev_cand=pct(candidate_recall), ev_cand20=pct(candidate_recall_20), ev_zs_r5=pct(zs_recall),
    ev_ft_r5=pct(ft_recall) if ft_recall is not None else '（沒有微調）',
    n_rerank_examples=f'{len(rerank_examples):,}', rerank_minutes=f'{rerank_minutes:.0f}',
    ev_given_doc=pct(evidence_given_doc), evidence_table=md_table(evidence_table),
    ev_recall_raw=evidence_recall, ev_dense_raw=dense_recall, ev_zs_raw=zs_recall,
    ev_ft_raw=ft_recall if ft_recall is not None else -1.0, ev_given_doc_raw=evidence_given_doc,
)
```

### 準備分類器的輸入文字，並存一份檢索快取

和 v1 的差別：
- **標準證據放最前面**（v1 是打亂）：test 時 reranker 也會把最像證據的句子排第 1，訓練和測試長得一樣；
  而且就算超過長度上限，被截掉的也是排在後面、比較不重要的句子。
- **每句最多 `SENTENCE_CHAR_LIMIT` 字**：少數超長的句子不會把其他句子擠出去。
- NEI 題的證據改用 reranker 挑的 5 句（上一格）。

其餘和 v1 相同：SUPPORTS / REFUTES 若標準證據不在檢索結果裡，就補進一組、擠掉排名最後的句子；找不到標準證據句的題目不拿來訓練。

**[Cell 36｜程式]**

```python
def format_sentence(key):
    """(頁名, 句號) → 「【頁名】句子」，句子最多 SENTENCE_CHAR_LIMIT 字。"""
    page_id, sentence_id = key
    sentence = get_sentence(page_id, sentence_id)
    return f'【{display_title(page_id)}】{sentence[:SENTENCE_CHAR_LIMIT]}' if sentence else ''


def build_context(keys):
    return ' '.join(part for part in map(format_sentence, keys) if part)


train_claims, train_contexts, train_label_ids = [], [], []
train_gold_examples = []  # (claim, 5 句, 標準證據) ── 第 4 節拿來量「截斷」問題
gold_injected = dropped = 0
for row, retrieved in zip(train_df.itertuples(), rows_to_keys(train_context_rows, EVIDENCE_TOP_K)):
    keys = list(retrieved)
    if row.label != NEI:
        usable = [group for group in gold_groups(row.evidence)
                  if all(get_sentence(page_id, sentence_id) for page_id, sentence_id in group)]
        if not usable:      # 標準證據句在維基裡找不到，這題沒辦法給正確證據，就不拿來訓練
            dropped += 1
            continue
        gold = sorted(min(usable, key=len))
        if not set(gold) <= set(keys):
            gold_injected += 1
        others = [key for key in keys if key not in gold]
        keys = gold + others[:max(0, EVIDENCE_TOP_K - len(gold))]  # 標準證據放最前面，不再打亂
        train_gold_examples.append((row.claim, keys, set(gold)))
    train_claims.append(row.claim)
    train_contexts.append(build_context(keys))
    train_label_ids.append(LABEL_TO_ID[row.label])

dev_contexts = [build_context(keys) for keys in dev_evidence]
test_contexts = [build_context(keys) for keys in test_evidence]
print(f'訓練資料 {len(train_contexts):,} 筆（補進標準證據 {gold_injected:,} 筆，找不到標準證據句而略過 {dropped:,} 筆）')
print('訓練範例：', train_claims[0], '\n          ', train_contexts[0][:200])
report_values.update(n_train_used=f'{len(train_contexts):,}', n_gold_injected=f'{gold_injected:,}', n_dropped=dropped)

retrieval_cache = {
    'dev_ids': dev_df['id'].tolist(), 'test_ids': test_df['id'].tolist(),
    'train_claims': train_claims, 'train_contexts': train_contexts, 'train_label_ids': train_label_ids,
    'dev_contexts': dev_contexts, 'test_contexts': test_contexts, 'dev_docs': dev_docs,
    'dev_evidence': dev_evidence, 'test_evidence': test_evidence,
    'dev_top_scores': dev_top_scores, 'test_top_scores': test_top_scores,
    'train_gold_examples': [[claim, keys, sorted(gold)] for claim, keys, gold in train_gold_examples],
    'report_values': report_values,
}
(OUTPUT_DIR / 'retrieval_cache.json').write_text(json.dumps(retrieval_cache, ensure_ascii=False), encoding='utf-8')
print('檢索結果已存到', OUTPUT_DIR / 'retrieval_cache.json')
```

**[Cell 37｜程式]**

```python
# 沿用 v2 的檢索結果（省約 40 分鐘），也是 Colab 斷線救援：
#   1. 執行第 1 節的三格、「讀取資料」，以及第 2 節第一格（gold_groups 等工具函式）
#   2. 把下面改成 True，按 Ctrl+Enter 執行這一格
#   3. 點下一格（第 4 節的標題），執行階段 → 執行之後的儲存格（Ctrl+F10）
# 正常 Run all 時保持 False，這一格什麼都不做。第 4 節的截斷分析需要讀維基句子，從快取載入時會自動跳過。
LOAD_RETRIEVAL_CACHE = False


def find_retrieval_cache():
    """先找 v3 自己存的，再找 v2 的（Colab：Drive 的 nlp-hw3/output_v2/；Kaggle：Input 裡的 retrieval_cache.json）。"""
    candidates = [OUTPUT_DIR / 'retrieval_cache.json', COLAB_DRIVE_DIR / 'output_v2' / 'retrieval_cache.json',
                  Path('output_v2') / 'retrieval_cache.json']
    if IN_KAGGLE:
        candidates += sorted(Path('/kaggle/input').rglob('retrieval_cache.json'))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('找不到 retrieval_cache.json。Colab：確認 Drive 的 nlp-hw3/output_v2/ 還在；'
                            'Kaggle：把 v2 的 output_v2 資料夾上傳成 Dataset，再用 Add Input 加進來')


if LOAD_RETRIEVAL_CACHE:
    cache_path = find_retrieval_cache()
    cache = json.loads(cache_path.read_text(encoding='utf-8'))
    assert cache['dev_ids'] == dev_df['id'].tolist() and cache['test_ids'] == test_df['id'].tolist(), \
        '資料筆數設定和快取不同，請重跑檢索'
    train_claims, train_contexts = cache['train_claims'], cache['train_contexts']
    train_label_ids, dev_contexts, test_contexts = cache['train_label_ids'], cache['dev_contexts'], cache['test_contexts']
    dev_docs, report_values = cache['dev_docs'], cache['report_values']
    dev_top_scores, test_top_scores = cache['dev_top_scores'], cache['test_top_scores']
    dev_evidence = [[tuple(key) for key in keys] for keys in cache['dev_evidence']]
    test_evidence = [[tuple(key) for key in keys] for keys in cache['test_evidence']]
    train_gold_examples = [(claim, [tuple(key) for key in keys], {tuple(key) for key in gold})
                           for claim, keys, gold in cache['train_gold_examples']]
    print('已載入檢索快取：', cache_path, f'（train {len(train_claims):,} 筆）')
```

## 4. Classification：3 個模型各自訓練，再 ensemble

每個模型的輸入都和 v2 一樣：句對 `claim` + `【頁名】句子…`。`truncation='only_second'` 表示太長時只截證據、不截 claim；
`DataCollatorWithPadding` 每一批只補到「這批最長」（dynamic padding）。

| 模型 | 參數量 | 和 v2 的模型差在哪 |
| :--- | ---: | :--- |
| `hfl/chinese-roberta-wwm-ext-large` | 326M | 就是 v2 的模型，重新訓練一次 |
| `hfl/chinese-macbert-large` | 326M | 同樣 24 層、同一套中文詞表；預訓練時要猜的字不是換成 [MASK]，而是換成意思相近的詞（MLM as correction），預訓練和 fine-tune 看到的文字比較像 |
| `FacebookAI/xlm-roberta-large` | 560M | 100 種語言一起預訓練，用 SentencePiece 斷詞，繁體字原生支援；詞表有 25 萬個，詞向量就佔了快一半參數，所以凍結詞向量層省記憶體 |

三個都只做過語言模型預訓練，沒有用其他標註資料集微調過。每個模型的 tokenizer 不一樣，所以 DataLoader 要在每個模型訓練前重建一次（下面的 `make_loaders`）。

**[Cell 39｜程式]**

```python
# 下面的截斷分析沿用 v2 的做法，只用第 1 個模型的 tokenizer 來量；真正訓練時每個模型會載入自己的 tokenizer
classifier_tokenizer = AutoTokenizer.from_pretrained(resolve_model(CLASSIFIER_MODELS[0]['name']))
```

### v1 的截斷問題有多嚴重？

從 train 的 SUPPORTS / REFUTES 抽 2,000 題，分別用 **v1 的設定**（256 token、順序打亂、句子不限長）和
**v2 的設定**（512 token、標準證據在前、每句 ≤ 120 字）組出輸入，檢查標準證據是不是**完整**落在長度上限之內。
被截掉的話，模型等於在「看不到證據」的情況下被要求回答 SUPPORTS / REFUTES，學到的是雜訊。

**[Cell 41｜程式]**

```python
def gold_kept_ratio(examples, max_length, char_limit, shuffle):
    """標準證據的最後一句有完整留在 max_length 之內的比例。"""
    rng = random.Random(SEED)
    kept = 0
    for claim, keys, gold in examples:
        keys = list(keys)
        if shuffle:
            rng.shuffle(keys)
        parts = []
        for page_id, sentence_id in keys:
            sentence = get_sentence(page_id, sentence_id) or ''
            parts.append(f'【{display_title(page_id)}】{sentence if char_limit is None else sentence[:char_limit]}')
        last_gold = max(i for i, key in enumerate(keys) if key in gold)  # 最後一句標準證據排在第幾個
        length = len(classifier_tokenizer(claim, ' '.join(parts[:last_gold + 1]))['input_ids'])
        kept += length <= max_length
    return kept / max(len(examples), 1)


if 'page_sentences' in globals() and train_gold_examples:
    sample = random.Random(SEED).sample(train_gold_examples, min(2000, len(train_gold_examples)))
    v1_kept = gold_kept_ratio(sample, 256, None, shuffle=True)
    v2_kept = gold_kept_ratio(sample, MAX_LENGTH, SENTENCE_CHAR_LIMIT, shuffle=False)
    lengths = [len(classifier_tokenizer(claim, build_context(keys))['input_ids']) for claim, keys, _ in sample]
    print(f'v1 設定：標準證據完整留在 256 token 內的比例 = {pct(v1_kept)}')
    print(f'v2 設定：標準證據完整留在 {MAX_LENGTH} token 內的比例 = {pct(v2_kept)}')
    print(f'v2 輸入長度中位數 {int(np.median(lengths))} token；超過 256 的有 {pct(np.mean([n > 256 for n in lengths]))}')
    report_values.update(trunc_v1=pct(v1_kept), trunc_v2=pct(v2_kept), median_len=int(np.median(lengths)),
                         over256=pct(np.mean([n > 256 for n in lengths])), trunc_v1_raw=v1_kept,
                         n_trunc_sample=len(sample))
else:
    print('從快取載入，跳過截斷分析')
```

**[Cell 42｜程式]**

```python
class ClaimDataset(Dataset):
    def __init__(self, tokenizer, claims, contexts, label_ids=None):
        self.encodings = tokenizer(list(claims), list(contexts), truncation='only_second', max_length=MAX_LENGTH)
        self.label_ids = label_ids

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, index):
        item = {name: values[index] for name, values in self.encodings.items()}
        if self.label_ids is not None:
            item['labels'] = self.label_ids[index]
        return item


def make_loaders(tokenizer):
    """每個模型的 tokenizer 不同，所以每個模型都要重新斷詞、重建 DataLoader。"""
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(ClaimDataset(tokenizer, train_claims, train_contexts, train_label_ids),
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
                              generator=torch.Generator().manual_seed(SEED))
    dev_loader = DataLoader(ClaimDataset(tokenizer, dev_df['claim'], dev_contexts),
                            batch_size=BATCH_SIZE * 4, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(ClaimDataset(tokenizer, test_df['claim'], test_contexts),
                             batch_size=BATCH_SIZE * 4, shuffle=False, collate_fn=collator)
    return train_loader, dev_loader, test_loader


# 同樣的 claim + 證據，3 個 tokenizer 切出來的長度不一樣（xlm-roberta 的 SentencePiece 常把幾個字併成一個 token）
for spec in CLASSIFIER_MODELS:
    tokenizer = AutoTokenizer.from_pretrained(resolve_model(spec['name']))
    encoded = tokenizer(dev_df['claim'][:200].tolist(), dev_contexts[:200], truncation='only_second', max_length=MAX_LENGTH)
    print(f"{spec['name']:36s} dev 前 200 題的輸入長度中位數：{int(np.median([len(ids) for ids in encoded['input_ids']]))} token")
```

### 訓練一個分類器（和 v2 一樣的 5 步驟，包成函式）

`train_classifier(spec)` 做的事和 v2 第 4 節完全一樣：AdamW、warmup 之後線性下降、gradient accumulation（每批 8 筆、累積 2 批 = 等效 batch 16）、
fp16、gradient clipping，每個 epoch 結束在 dev 上算 FEVER score，留最好的 epoch。差別只有三個：

- 模型、tokenizer、學習率都從 `spec` 來，3 個模型共用同一個函式；
- `freeze_embeddings=True` 時凍結詞向量層（和 v2 微調 reranker 一樣），xlm-roberta-large 的 25 萬詞表就不用存梯度和 Adam 狀態；
- 訓練完只回傳 dev / test 的 **logits**，模型本身用完就刪掉，下一個模型才有 GPU 記憶體可以用。

**[Cell 44｜程式]**

```python
@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    outputs = []
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
            outputs.append(model(**batch).logits.float().cpu())
    return torch.cat(outputs)


def logits_to_labels(logits, nei_bias=0.0):
    adjusted = logits.clone()
    adjusted[:, NEI_ID] += nei_bias  # 第 5 節會用到：NEI 的分數加一點 / 減一點
    return [LABELS[index] for index in adjusted.argmax(dim=1).tolist()]


def label_accuracy(df, predicted_labels):
    return float(np.mean([predicted == gold for predicted, gold in zip(predicted_labels, df['label'])]))


def fever_score(df, predicted_labels, predicted_evidence):
    """標籤對，而且（NEI 以外）前 5 句完整命中一組標準證據，才算這題對。"""
    hits = 0
    for gold_label, evidence, label, keys in zip(df['label'], df['evidence'], predicted_labels, predicted_evidence):
        if label == gold_label and (gold_label == NEI or evidence_hit(gold_groups(evidence), keys[:5])):
            hits += 1
    return hits / max(len(df), 1)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_classifier(spec):
    """訓練一個分類器；回傳 dev / test 的 logits、每個 epoch 的紀錄、參數量和採用的 epoch。"""
    set_seed(SEED)
    path = resolve_model(spec['name'])
    tokenizer = AutoTokenizer.from_pretrained(path)
    train_loader, dev_loader, test_loader = make_loaders(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
    ).float().to(device)  # 權重一律用 fp32 存，fp16 只在 autocast 裡用（GradScaler 不能處理 fp16 的權重）
    if spec['freeze_embeddings']:
        for parameter in model.base_model.embeddings.parameters():
            parameter.requires_grad = False  # 凍結詞向量層
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = count_parameters(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"\n===== {spec['name']}：{params:,} 參數（可訓練 {sum(p.numel() for p in trainable):,}），lr = {spec['lr']} =====")

    optimizer = torch.optim.AdamW(trainable, lr=spec['lr'], weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_steps = updates_per_epoch * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_FP16)  # fp16 訓練時把 loss 放大，避免梯度太小變成 0
    print(f'每個 epoch {len(train_loader):,} 批、更新 {updates_per_epoch:,} 次；總共更新 {total_steps:,} 次')

    epoch_rows = []
    best_fever, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()                                                    # 1. 清掉梯度
        total_loss, start_time = 0.0, time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_FP16):
                loss = model(**batch).loss                                       # 2. 前向 + cross entropy
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()                     # 3. 反向傳播，梯度會一直累加
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):       # 累積夠了才更新
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)          # 4. 梯度裁剪
                scaler.step(optimizer)                                           # 5. 更新參數
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()                                            # 下一輪累積前清掉
            total_loss += loss.item()
            if step % 400 == 0 or step == len(train_loader):
                print(f'Epoch {epoch} step {step}/{len(train_loader)}  平均 loss = {total_loss / step:.4f}'
                      f'  （{time.time() - start_time:.0f} 秒）')
        epoch_labels = logits_to_labels(predict_logits(model, dev_loader))
        epoch_fever = fever_score(dev_df, epoch_labels, dev_evidence)
        epoch_rows.append({'epoch': epoch, 'train loss': total_loss / len(train_loader),
                           'dev Accuracy': label_accuracy(dev_df, epoch_labels), 'dev FEVER score': epoch_fever})
        print(f"epoch {epoch}：dev Accuracy {pct(epoch_rows[-1]['dev Accuracy'])}，dev FEVER score {pct(epoch_fever)}")
        if epoch_fever > best_fever:
            best_fever, best_epoch = epoch_fever, epoch
            if epoch < EPOCHS:  # 最後一個 epoch 就是現在的權重，不用另外存
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_epoch != EPOCHS:
        model.load_state_dict(best_state)
        print(f'第 {best_epoch} 個 epoch 的 dev FEVER score 最高，改回那時候的權重')

    result = {'name': spec['name'], 'params': params, 'lr': spec['lr'], 'freeze_embeddings': spec['freeze_embeddings'],
              'best_epoch': best_epoch, 'epoch_rows': epoch_rows,
              'dev_logits': predict_logits(model, dev_loader).tolist(),
              'test_logits': predict_logits(model, test_loader).tolist()}
    del model, optimizer, scheduler, scaler, best_state, trainable
    free_gpu()
    return result
```

### 依序訓練 3 個模型（每訓練完一個就存檔）

每個模型訓練完，dev / test 的 logits 會存成 `OUTPUT_DIR/pred_模型名.json`。Colab 中途斷線的話，重新跑到這一格時，
已經存好的模型會直接讀檔、不再重訓（`REUSE_SAVED_PREDICTIONS = True`）。

檔案裡也記了一個「輸入指紋」：dev / test 的 claim 和證據文字算出來的短碼。如果檢索結果變了（例如重跑了檢索），
或是改了這個模型的學習率、epoch 數，對不上就會自動重訓，不會拿舊的預測來混。

**[Cell 46｜程式]**

```python
def input_fingerprint():
    """dev / test 的 claim + 證據文字 → 一串短碼；檢索結果有變，短碼就會不一樣。"""
    text = '\n'.join(dev_df['claim'].tolist() + list(dev_contexts) + test_df['claim'].tolist() + list(test_contexts))
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def same_setting(saved, spec):
    return (saved.get('fingerprint') == fingerprint and saved.get('epochs') == EPOCHS
            and saved.get('lr') == spec['lr'] and saved.get('freeze_embeddings') == spec['freeze_embeddings'])


fingerprint = input_fingerprint()
classifier_results = []
for spec in CLASSIFIER_MODELS:
    pred_path = OUTPUT_DIR / f"pred_{spec['name'].split('/')[-1]}.json"
    if REUSE_SAVED_PREDICTIONS and pred_path.is_file():
        saved = json.loads(pred_path.read_text(encoding='utf-8'))
        if same_setting(saved, spec):
            print(f"讀取已經訓練好的 {spec['name']}：{pred_path}")
            classifier_results.append(saved)
            continue
        print(f'{pred_path.name} 的輸入或設定和這次不同，重新訓練')
    start_time = time.time()
    result = train_classifier(spec)
    result.update(fingerprint=fingerprint, epochs=EPOCHS, minutes=round((time.time() - start_time) / 60, 1))
    pred_path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    print(f"{spec['name']} 訓練完成，花了 {result['minutes']} 分鐘，預測已存到 {pred_path}")
    classifier_results.append(result)

# 學不起來的模型（dev Accuracy 卡在 45% 以下，large 模型偶爾會這樣）不放進 ensemble
for result in classifier_results:
    result['collapsed'] = max(row['dev Accuracy'] for row in result['epoch_rows']) < 0.45
    if result['collapsed']:
        print(f"⚠️ {result['name']} 的 dev Accuracy 很低，沒有學起來，不放進 ensemble。"
              '可以把它的 lr 調小（例如 5e-6）再重跑這一格（設定變了會自動重訓）')
usable_results = [result for result in classifier_results if not result['collapsed']]
assert usable_results, '3 個模型都沒有學起來，請檢查學習率'
print(f'可以用的模型：{len(usable_results)} / {len(classifier_results)} 個')
```

## 5. Validation：Ensemble、Accuracy、FEVER score 與 NEI 決策

**Ensemble**：每個模型的 logits 先做 softmax 變成三類機率，幾個模型的機率取平均，再取 log 當作新的分數。
取 log 是為了讓後面的 NEI bias 和 v2 一樣「加在分數上」：對單一模型來說，log softmax 和原本的 logits 只差一個常數，取 argmax 的結果完全一樣。

所有模型組合（單一模型、兩兩一組、三個一起）都在 dev 上試一次，每個組合各自挑最好的 NEI bias 和門檻，
最後用 dev FEVER score 最高的組合（同分時選模型多的，比較穩）。

**Accuracy** 只比對標籤；**FEVER score** 還要求 SUPPORTS / REFUTES 的前 5 句證據完整命中（NEI 只看標籤）。決策的兩個旋鈕和 v2 一樣：

1. **NEI bias**：NEI 的分數加一個數字，再取 argmax。
2. **reranker 門檻**：reranker 給「第 1 名句子」的分數低於門檻，代表連最像的句子都不太像證據，直接判 NEI。
   門檻取 dev 上第 1 名分數的分位數，例如 10% = 分數最低的那 10% 題目一律判 NEI。

**[Cell 48｜程式]**

```python
def ensemble_scores(results, split):
    """幾個模型的三類機率取平均，再取 log。split = 'dev' 或 'test'。"""
    probabilities = torch.stack([torch.softmax(torch.tensor(result[f'{split}_logits']), dim=1) for result in results])
    return probabilities.mean(dim=0).clamp_min(1e-9).log()  # [題數, 3]


def decide_labels(logits, top_scores, nei_bias, threshold):
    """先用 NEI bias 決定標籤；有門檻時，reranker 第 1 名分數低於門檻的題目改判 NEI。"""
    labels = logits_to_labels(logits, nei_bias)
    if threshold is None:
        return labels
    return [NEI if score < threshold else label for label, score in zip(labels, top_scores)]


def quantile_text(q):
    return f'{q * 100:g}%'


threshold_options = [(q, None if q == 0 else float(np.quantile(dev_top_scores, q))) for q in SCORE_QUANTILES]


def search_rules(dev_scores):
    """NEI bias × reranker 門檻的所有組合都在 dev 上試一次。"""
    grid = []
    for q, threshold in threshold_options:
        for bias in NEI_BIAS_GRID:
            labels = decide_labels(dev_scores, dev_top_scores, bias, threshold)
            grid.append({'q': q, 'threshold': threshold, 'bias': bias,
                         'fever': fever_score(dev_df, labels, dev_evidence), 'acc': label_accuracy(dev_df, labels),
                         'nei_ratio': labels.count(NEI) / len(labels)})
    return grid


def pick_rule(grid):
    """FEVER 最高的組合；同分時先選不用門檻（q 小）、再選 bias 離 0 近的，比較不容易只是剛好適合 dev。"""
    return max(grid, key=lambda row: (round(row['fever'], 6), -row['q'], -abs(row['bias'])))


def short_name(result):
    return result['name'].split('/')[-1]


# ---- 所有模型組合（單一、兩兩、全部）都在 dev 上比一次 ----
combo_rows = []
for size in range(1, len(usable_results) + 1):
    for combo in combinations(usable_results, size):
        scores = ensemble_scores(combo, 'dev')
        plain = decide_labels(scores, dev_top_scores, 0.0, None)
        combo_rows.append({'combo': combo, 'rule': pick_rule(search_rules(scores)),
                           'acc_plain': label_accuracy(dev_df, plain), 'fever_plain': fever_score(dev_df, plain, dev_evidence)})
best_combo_row = max(combo_rows, key=lambda row: (round(row['rule']['fever'], 6), len(row['combo'])))  # 同分選模型多的
chosen_results = list(best_combo_row['combo'])
chosen_combo = ' + '.join(short_name(result) for result in chosen_results)
ensemble_table = pd.DataFrame([{
    '模型組合': ' + '.join(short_name(result) for result in row['combo']),
    'Accuracy（不調整）': pct(row['acc_plain']), 'FEVER（不調整）': pct(row['fever_plain']),
    '最好的 NEI bias': row['rule']['bias'],
    'reranker 門檻': '不使用' if row['rule']['threshold'] is None else f"最低 {quantile_text(row['rule']['q'])}",
    'FEVER（調整後）': pct(row['rule']['fever']),
} for row in combo_rows])
print('所有模型組合：')
print(md_table(ensemble_table))
print('選用的組合：', chosen_combo)

# ---- 選好的組合：和 v2 一樣列出 bias 表和門檻表 ----
dev_logits = ensemble_scores(chosen_results, 'dev')
grid = search_rules(dev_logits)
best = pick_rule(grid)
best_bias, best_threshold, best_q = best['bias'], best['threshold'], best['q']

bias_table = pd.DataFrame([{'NEI bias': row['bias'], 'Accuracy': pct(row['acc']), 'FEVER score': pct(row['fever']),
                            '預測為 NEI 的比例': pct(row['nei_ratio'])} for row in grid if row['q'] == 0])
rule_rows = []
for q, threshold in threshold_options:
    row = max((r for r in grid if r['q'] == q), key=lambda r: (round(r['fever'], 6), -abs(r['bias'])))
    rule_rows.append({'reranker 門檻': '不使用' if threshold is None else f'最低 {quantile_text(q)}（分數 < {threshold:.2f}）',
                      '該門檻下最好的 NEI bias': row['bias'], 'Accuracy': pct(row['acc']),
                      'FEVER score': pct(row['fever']), '預測為 NEI 的比例': pct(row['nei_ratio'])})
rule_table = pd.DataFrame(rule_rows)
print('\n只調 NEI bias：')
print(md_table(bias_table))
print('\nNEI bias ＋ reranker 門檻：')
print(md_table(rule_table))
print(f'選用：NEI bias = {best_bias}，門檻 = {"不使用" if best_threshold is None else f"{best_threshold:.3f}（最低 {quantile_text(best_q)}）"}')

dev_labels_plain = decide_labels(dev_logits, dev_top_scores, 0.0, None)
dev_labels_bias_only = decide_labels(dev_logits, dev_top_scores,
                                     max((r for r in grid if r['q'] == 0), key=lambda r: r['fever'])['bias'], None)
dev_labels = decide_labels(dev_logits, dev_top_scores, best_bias, best_threshold)
accuracy_plain, accuracy_best = label_accuracy(dev_df, dev_labels_plain), label_accuracy(dev_df, dev_labels)
fever_plain = fever_score(dev_df, dev_labels_plain, dev_evidence)
fever_bias_only = fever_score(dev_df, dev_labels_bias_only, dev_evidence)
fever_best = fever_score(dev_df, dev_labels, dev_evidence)
print(f'Classification Accuracy: {accuracy_best:.4f}（不調整時 {accuracy_plain:.4f}）')
print(f'FEVER score（本地估計）: {fever_best:.4f}（不調整 {fever_plain:.4f}；只調 bias {fever_bias_only:.4f}）')

confusion = pd.crosstab(pd.Series(dev_df['label'], name='正確答案'), pd.Series(dev_labels, name='預測'))
confusion = confusion.reindex(index=LABELS, columns=LABELS, fill_value=0)
confusion_table = confusion.reset_index().rename(columns={'正確答案': '正確答案 \\ 預測'})
print(md_table(confusion_table))

# 三階段漏斗：可驗證題一路追下去
funnel = Counter()
for label, evidence, docs, keys, predicted in zip(dev_df['label'], dev_df['evidence'], dev_docs, dev_evidence, dev_labels):
    if not is_verifiable(label):
        continue
    groups = gold_groups(evidence)
    funnel['total'] += 1
    if doc_hit(groups, docs):
        funnel['doc'] += 1
        if evidence_hit(groups, keys):
            funnel['evidence'] += 1
            funnel['label'] += predicted == label
nei_total = int((dev_df['label'] == NEI).sum())
nei_correct = sum(predicted == NEI for label, predicted in zip(dev_df['label'], dev_labels) if label == NEI)
total = max(funnel['total'], 1)
funnel_table = pd.DataFrame([
    {'階段': 'SUPPORTS / REFUTES 題數', '題數': funnel['total'], '佔可驗證題': '100.00%'},
    {'階段': f'① 前 {DOCUMENT_TOP_K} 頁找齊證據頁', '題數': funnel['doc'], '佔可驗證題': pct(funnel['doc'] / total)},
    {'階段': f'② 前 {EVIDENCE_TOP_K} 句找齊證據句', '題數': funnel['evidence'], '佔可驗證題': pct(funnel['evidence'] / total)},
    {'階段': '③ 標籤也判斷正確（= 拿到 FEVER 分數）', '題數': funnel['label'], '佔可驗證題': pct(funnel['label'] / total)},
])
print(md_table(funnel_table))
print(f'NOT ENOUGH INFO：{nei_correct} / {nei_total} 題判斷正確')

# 證據有找齊的題目裡，標準證據排在 reranker 第 1 句的比例（檢查訓練時「標準證據放最前面」和 test 是否一致）
gold_found = gold_top1 = 0
for label, evidence, keys in zip(dev_df['label'], dev_df['evidence'], dev_evidence):
    if not is_verifiable(label):
        continue
    keys = [(page_id, int(sentence_id)) for page_id, sentence_id in keys]
    found = [group for group in gold_groups(evidence) if group <= set(keys)]
    if found and keys:
        gold_found += 1
        gold_top1 += any(keys[0] in group for group in found)
print(f'證據有找齊的 {gold_found} 題中，標準證據排在第 1 句的有 {gold_top1} 題（{pct(gold_top1 / max(gold_found, 1))}）')

# 每個模型單獨的表現 + 每個 epoch 的紀錄（報告第 3 題用）
single_rows = {id(row['combo'][0]): row for row in combo_rows if len(row['combo']) == 1}
classifier_rows, epoch_parts = [], []
for result in classifier_results:
    best_row = result['epoch_rows'][result['best_epoch'] - 1]
    single = single_rows.get(id(result))
    classifier_rows.append({
        '模型': short_name(result), '參數量': f"{result['params'] / 1e6:.0f}M", 'learning rate': result['lr'],
        '凍結詞向量': '是' if result['freeze_embeddings'] else '否', '採用的 epoch': result['best_epoch'],
        'dev Accuracy': pct(best_row['dev Accuracy']), 'dev FEVER（不調整）': pct(best_row['dev FEVER score']),
        'dev FEVER（調 bias / 門檻）': pct(single['rule']['fever']) if single else '沒學起來，不採用',
    })
    epoch_parts.append(f"`{result['name']}`：\n\n" + md_table(pd.DataFrame([
        {'epoch': row['epoch'], 'train loss': f"{row['train loss']:.4f}", 'dev Accuracy': pct(row['dev Accuracy']),
         'dev FEVER score': pct(row['dev FEVER score'])} for row in result['epoch_rows']])))
classifier_table = pd.DataFrame(classifier_rows)
print(md_table(classifier_table))
best_single = max((row for row in combo_rows if len(row['combo']) == 1), key=lambda row: row['rule']['fever'])

report_values.update(
    best_bias=best_bias, best_q=quantile_text(best_q),
    best_threshold='不使用' if best_threshold is None else f'{best_threshold:.3f}',
    acc_best=pct(accuracy_best), acc_plain=pct(accuracy_plain), fever_best=pct(fever_best),
    fever_plain=pct(fever_plain), fever_bias_only=pct(fever_bias_only), bias_table=md_table(bias_table),
    rule_table=md_table(rule_table), confusion_table=md_table(confusion_table), funnel_table=md_table(funnel_table),
    nei_correct=nei_correct, nei_total=nei_total,
    funnel_drops={'文件檢索': funnel['total'] - funnel['doc'], '證據檢索': funnel['doc'] - funnel['evidence'],
                  '分類': funnel['evidence'] - funnel['label']},
    funnel_total=funnel['total'], funnel_label=funnel['label'], acc_best_raw=accuracy_best,
    acc_plain_raw=accuracy_plain, fever_best_raw=fever_best, fever_plain_raw=fever_plain,
    fever_bias_only_raw=fever_bias_only, best_q_raw=best_q,
    gold_top1_ratio=pct(gold_top1 / max(gold_found, 1)),
    classifier_table=md_table(classifier_table), epoch_tables='\n\n'.join(epoch_parts),
    ensemble_table=md_table(ensemble_table), chosen_combo=chosen_combo, chosen_size=len(chosen_results),
    n_models=len(classifier_results), best_single_name=short_name(best_single['combo'][0]),
    best_single_fever=pct(best_single['rule']['fever']), best_single_fever_raw=best_single['rule']['fever'],
    classifier_params_text='、'.join(f"{short_name(r)}（{r['params'] / 1e6:.0f}M）" for r in classifier_results),
)
```

**[Cell 49｜程式]**

```python
# 錯誤分析：各挑幾題來看（寫報告第 4 題時可以拿來舉例）
def show_examples(title, condition, limit=3):
    print(f'\n===== {title} =====')
    shown = 0
    for i, row in dev_df.iterrows():
        if shown < limit and condition(i, row):
            gold_pages = sorted({page for group in gold_groups(row['evidence']) for page, _ in group})
            print(f"[{row['label']} → 預測 {dev_labels[i]}] {row['claim']}")
            print(f'   標準證據頁：{gold_pages}\n   取回頁面：  {dev_docs[i]}')
            shown += 1


show_examples('文件檢索失敗', lambda i, row: is_verifiable(row['label'])
              and not doc_hit(gold_groups(row['evidence']), dev_docs[i]))
show_examples('頁面對了、句子沒找齊', lambda i, row: is_verifiable(row['label'])
              and doc_hit(gold_groups(row['evidence']), dev_docs[i])
              and not evidence_hit(gold_groups(row['evidence']), dev_evidence[i]))
show_examples('證據找齊、標籤判斷錯', lambda i, row: is_verifiable(row['label'])
              and evidence_hit(gold_groups(row['evidence']), dev_evidence[i]) and dev_labels[i] != row['label'])
```

## 6. 產生 submission.csv

三欄：`id`、`predicted_label`（supports / refutes / NOT ENOUGH INFO）、`predicted_evidence`（JSON 字串，NEI 填 `[]`）。
順序和 `test.jsonl` 完全相同；test 用的模型組合、NEI bias 和門檻都是第 5 節在 dev 上選出來的。

**[Cell 51｜程式]**

```python
test_logits = ensemble_scores(chosen_results, 'test')
test_labels = decide_labels(test_logits, test_top_scores, best_bias, best_threshold)
rows = []
for claim_id, label, keys in zip(test_df['id'], test_labels, test_evidence):
    evidence = [] if label == NEI else [[page_id, int(sentence_id)] for page_id, sentence_id in keys[:EVIDENCE_TOP_K]]
    rows.append({'id': claim_id, 'predicted_label': SUBMIT_LABEL[label],
                 'predicted_evidence': json.dumps(evidence, ensure_ascii=False)})
submission = pd.DataFrame(rows)
assert len(submission) == len(test_df)
assert submission['id'].tolist() == test_df['id'].tolist()
assert submission['predicted_label'].isin(SUBMIT_LABEL.values()).all()
if TEST_SAMPLES is None:
    submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False, encoding='utf-8')
    print('已寫出', OUTPUT_DIR / 'submission.csv', f'（{len(submission)} 筆，模型組合：{chosen_combo}）')
else:
    print('⚠️ TEST_SAMPLES 不是 None（試跑模式），沒有寫出 submission.csv')
print('test 預測分布：', dict(Counter(test_labels)))
submission.head()
```

## 7. 報告自動填空

下面這格會把這次執行的數字填進報告，印出來並存成 `report_filled.md`。
**把印出來的內容整段貼到最下面的 Report 格子**（題目和格式沒有改，只填 Answer），再補上 Kaggle 分數即可。
v1 / v2 / v3 對照表裡 v1、v2 的數字來自第 1 節的 `PREVIOUS_RESULTS`（前兩版實際跑出來的結果）。

**[Cell 53｜程式]**

```python
v = dict(report_values)
prev = PREVIOUS_RESULTS


def compare(new, old, unit='個百分點'):
    diff = (new - old) * 100
    if abs(diff) < 0.5:
        return f'幾乎一樣（差 {diff:+.2f} {unit}）'
    return f'{"提升" if diff > 0 else "下降"}了 {abs(diff):.2f} {unit}'


# ---- v1 / v2 / v3 對照（v1、v2 的數字在第 1 節的 PREVIOUS_RESULTS）----
version_table = pd.DataFrame([
    {'指標（dev）': f'Document Recall@{DOCUMENT_TOP_K}', 'v1': prev['v1']['doc_r5'], 'v2': prev['v2']['doc_r5'], 'v3': v['doc_r5']},
    {'指標（dev）': f'Evidence Recall@{EVIDENCE_TOP_K}', 'v1': prev['v1']['ev_r5'], 'v2': prev['v2']['ev_r5'], 'v3': v['ev_r5']},
    {'指標（dev）': 'Classification Accuracy', 'v1': prev['v1']['acc_best'], 'v2': prev['v2']['acc_best'], 'v3': v['acc_best']},
    {'指標（dev）': 'FEVER score', 'v1': prev['v1']['fever_best'], 'v2': prev['v2']['fever_best'], 'v3': v['fever_best']},
    {'指標（dev）': 'Kaggle public score', 'v1': prev['v1']['public'], 'v2': prev['v2']['public'], 'v3': '（提交後填入）'},
])
v['version_table'] = md_table(version_table)
same_retrieval = v['doc_r5'] == prev['v2']['doc_r5'] and v['ev_r5'] == prev['v2']['ev_r5']
v['version_compare_text'] = (
    ('v3 沿用 v2 的檢索結果，文件和證據兩段的數字完全一樣，所以 FEVER score 的差別全部來自分類。' if same_retrieval else
     '這次重新跑了檢索，文件和證據的數字和 v2 有一點差異（reranker 的微調在 GPU 上不是完全可重現）。')
    + f"dev FEVER score 從 v2 的 {prev['v2']['fever_best']} 變成 {v['fever_best']}，{compare(v['fever_best_raw'], prev['v2']['fever_raw'])}。"
)

if v['chosen_size'] == 1:
    v['ensemble_text'] = (f"這次 ensemble 在 dev 上沒有比最好的單一模型 {v['best_single_name']}（{v['best_single_fever']}）好，"
                          '所以 test 只用這一個模型。')
else:
    v['ensemble_text'] = (f"單一模型裡最好的是 {v['best_single_name']}（調整後 dev FEVER score {v['best_single_fever']}），"
                          f"選用的組合是 {v['fever_best']}，{compare(v['fever_best_raw'], v['best_single_fever_raw'])}。")

best_single_doc = max(v['doc_title_only_raw'], v['doc_dense_only_raw'])
v['doc_compare_text'] = (
    f"只用標題比對的 Recall@{DOCUMENT_TOP_K} 是 {v['doc_title_only']}，只用向量是 {v['doc_dense_only']}，"
    f"兩種合起來是 {v['doc_r5']}，比單用較好的那一種{compare(v['doc_recall_raw'], best_single_doc)}。"
)
if v['best_bonus'] == 0:
    v['bonus_text'] = '結果 bonus = 0 就是最好的，代表 bge-m3 自己已經會把標題命中的頁排在前面，額外加分沒有幫助。'
elif v['doc_max_bonus_raw'] >= v['doc_recall_raw'] - 1e-9:
    v['bonus_text'] = (f"只要給標題命中的頁一點加分（bonus = {v['best_bonus']}），Recall@{DOCUMENT_TOP_K} 就從 "
                       f"{v['doc_dense_only']} 變成 {v['doc_r5']}，再把 bonus 加大到 {max(TITLE_BONUS_GRID)} 也一樣是 "
                       f"{v['doc_max_bonus']}。這代表「claim 裡出現的頁名」是非常強的線索，標題命中的頁幾乎都該排在前面。")
else:
    v['bonus_text'] = (f"bonus = {v['best_bonus']} 時 Recall@{DOCUMENT_TOP_K} 最高（{v['doc_r5']}），"
                       f"但 bonus 開到 {max(TITLE_BONUS_GRID)} 時下降到 {v['doc_max_bonus']}。我推測是因為國名、年份這類很通用的短標題"
                       '也會被比對到，加太多分會把真正的頁擠出前 5 名，所以標題比對適合當「加分」而不是「絕對優先」。')

if v['ev_ft_raw'] >= 0:
    ft_effect = compare(v['ev_ft_raw'], v['ev_zs_raw'])
    v['evidence_compare_text'] = (
        f"只用向量挑 5 句的 Recall 是 {v['ev_dense_r5']}；現成的 reranker 從 {SENTENCE_CANDIDATES} 句重排後是 {v['ev_zs_r5']}；"
        f"用 train 微調 reranker 之後是 {v['ev_ft_r5']}，比現成的{ft_effect}，所以最後採用{v['reranker_choice']} reranker。"
        f"向量前 {SENTENCE_CANDIDATES} 句的 Recall 是 {v['ev_cand']}（前 20 句是 {v['ev_cand20']}），這是 reranker 再厲害也超不過的上限，"
        f"這也是我把初篩從 v1 的 20 句放寬到 {SENTENCE_CANDIDATES} 句的原因。"
    )
else:
    v['evidence_compare_text'] = (f"只用向量挑 5 句的 Recall 是 {v['ev_dense_r5']}，加上現成 reranker 之後是 {v['ev_zs_r5']}，"
                                  f"{compare(v['ev_zs_raw'], v['ev_dense_raw'])}。")
if v['ev_given_doc_raw'] >= 0.75:
    v['evidence_bottleneck_text'] = '只要頁面找對，大部分題目都能把證據句找齊，證據 Recall 的損失主要來自第 1 階段沒有找到正確頁面。'
else:
    v['evidence_bottleneck_text'] = '就算頁面已經找對，挑句子這一步還是會漏掉不少，特別是需要 2 句以上一起才能判斷的題目。'

trunc_note = ''
if 'trunc_v1' not in v:  # 從快取載入時沒有維基句子可以重量，引用 v2 量到的數字（輸入格式和 v2 相同）
    v.update({key: prev['v2'][key] for key in ('trunc_v1', 'trunc_v2', 'median_len', 'over256', 'trunc_v1_raw', 'n_trunc_sample')})
    trunc_note = '（這組數字是 v2 實際量的，v3 的輸入格式和 v2 相同。）'
if v['trunc_v1_raw'] < 0.9:
    trunc_conclusion = ('所以 v1 確實有不少訓練資料是在「看不到證據」的情況下被要求回答 SUPPORTS / REFUTES，'
                        '模型學到的其實是雜訊，這是 v2 把長度放寬、標準證據放前面的原因。')
else:
    trunc_conclusion = '所以 v1 的截斷影響其實不大，v2 的進步主要來自 reranker 和分類器的改動；放寬長度主要是讓 5 句證據都能被看到。'
v['trunc_text'] = (f"我從 train 的 SUPPORTS / REFUTES 抽 {v['n_trunc_sample']:,} 題實際量過：v1 的設定（256 token、證據順序打亂、句子不限長）下，"
                   f"標準證據能完整留在輸入裡的比例是 {v['trunc_v1']}；v2 的設定（{MAX_LENGTH} token、標準證據在前、每句 ≤ "
                   f"{SENTENCE_CHAR_LIMIT} 字）是 {v['trunc_v2']}。輸入長度的中位數是 {v['median_len']} token，"
                   f"超過 256 的有 {v['over256']}，" + trunc_conclusion + trunc_note)

drops = v['funnel_drops']
worst = max(drops, key=drops.get)
advice = {
    '文件檢索': '代表還有不少正確頁面沒被找到（例如 claim 用的是別名、簡稱，頁名沒有直接出現），之後可以讓頁面也用 cross-encoder 重排。',
    '證據檢索': '代表頁面找對了但句子沒挑齊，常見的是需要 2 句一起才能判斷的題目，之後可以讓 reranker 一次看多句（multi-hop）。',
    '分類': '代表證據已經找到但分類器判斷錯。v3 已經用了 ensemble，之後可以把 train + dev 合起來訓練最終模型，或換更大的分類模型。',
}
v['funnel_text'] = (f"損失最多的是「{worst}」這一步，少了 {drops[worst]} 題（佔可驗證題 {pct(drops[worst] / max(v['funnel_total'], 1))}），"
                    + advice[worst] + f"另外，NOT ENOUGH INFO 的 {v['nei_total']} 題中有 {v['nei_correct']} 題判斷正確。")
v['funnel_compare_text'] = (f"和 v2 比：第 ③ 步（證據找齊、標籤也對）從 {prev['v2']['label_correct']} 題變成 {v['funnel_label']} 題，"
                            f"NOT ENOUGH INFO 判對的題數從 {prev['v2']['nei_correct']} 題變成 {v['nei_correct']} 題。")

fever_vs_accuracy = ('Accuracy 和 FEVER score 看的東西不一樣：SUPPORTS / REFUTES 的題目如果證據沒找到，標籤猜對也拿不到分，'
                     '把這種題改猜 NEI 幾乎沒有損失，卻能多拿到真正 NEI 題的分數。')
if v['best_q_raw'] == 0 and v['best_bias'] == 0:
    v['rule_text'] = f"結果兩個旋鈕都不調時 FEVER score 就最高（{v['fever_best']}），所以 test 直接用分類器的原始預測。"
elif v['best_q_raw'] == 0:
    v['rule_text'] = (
        f"最後選的是 NEI bias = {v['best_bias']}，沒有用 reranker 門檻（加上門檻在 dev 上沒有更好，代表分類器自己已經能從證據看出「找不到」）。"
        f"dev FEVER score 從不調整的 {v['fever_plain']} 變成 {v['fever_best']}，Accuracy 則從 {v['acc_plain']} 變成 {v['acc_best']}。"
        + fever_vs_accuracy + '因為競賽是用 FEVER score 排名，test 我用這組設定。'
    )
else:
    v['rule_text'] = (
        f"最後選的是 NEI bias = {v['best_bias']}，而且 reranker 第 1 名分數最低的 {v['best_q']} 題（分數 < {v['best_threshold']}）一律判 NEI。"
        f"dev FEVER score 從不調整的 {v['fever_plain']}，只調 bias 變成 {v['fever_bias_only']}，兩個一起調變成 {v['fever_best']}；"
        f"Accuracy 則從 {v['acc_plain']} 變成 {v['acc_best']}。" + fever_vs_accuracy
        + 'reranker 的分數正好是「證據有沒有找到」的訊號，所以比只看分類器的 NEI 機率多一個判斷依據。'
        '因為競賽是用 FEVER score 排名，test 我用這組設定。'
    )

v.update(
    max_title_pages=MAX_TITLE_PAGES, page_lead_chars=PAGE_LEAD_CHARS, dense_candidates=DENSE_CANDIDATES,
    bonus_grid=', '.join(map(str, TITLE_BONUS_GRID)), doc_k=DOCUMENT_TOP_K, ev_k=EVIDENCE_TOP_K,
    sentence_candidates=SENTENCE_CANDIDATES, max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM_STEPS, effective_batch=BATCH_SIZE * GRAD_ACCUM_STEPS,
    epochs=EPOCHS, warmup_pct=f'{WARMUP_RATIO:.0%}', rerank_negatives=RERANK_NEGATIVES, rerank_lr=RERANK_LR,
    rerank_groups=RERANK_TRAIN_GROUPS, sentence_char_limit=SENTENCE_CHAR_LIMIT,
    nei_candidates=RERANK_TRAIN_NEI_CANDIDATES, v1_public=prev['v1']['public'], v2_public=prev['v2']['public'],
    v1_rank=prev['v1']['rank'], v2_rank=prev['v2']['rank'], doc_miss=pct(1 - v['doc_recall_raw']),
    bias_grid=', '.join(map(str, NEI_BIAS_GRID)),
    quantile_grid=', '.join(quantile_text(q) for q in SCORE_QUANTILES if q > 0),
)

REPORT_TEMPLATE = r'''# Report

Your Kaggle username: zhi816

Your public leaderboard score: （提交 Kaggle 後填入）

Your private leaderboard score: （11/23 公布後填入）


1. Describe your method for retrieving Wikipedia documents (5%) and report Recall@N on development set (5%).

Answer:

我讀 sample code 時發現，它只把 train 有標註過的頁面當候選，dev / test 的正確頁面如果沒在 train 出現過，就一定找不到。所以我改成讀完整的 `wiki_pages.jsonl`（共 {n_wiki_pages} 頁），並把文件檢索拆成「先找候選、再排順序」兩步（這一段 v1、v2、v3 三個版本都相同）：

* **(1) 標題比對（找候選）**：維基的頁名常常就是 claim 裡的主角，例如 claim 在講某部電影，頁名通常就是那部電影的名字。我先把所有頁名正規化（去掉結尾的消歧義括號、統一「·」這類間隔號、去掉空白、英文轉小寫），做成「標題 → 頁面」的 dict，再把 claim 中所有長度 2～30 的子字串拿去查。查到的就是候選頁，長的標題比較具體所以排前面，每題最多留 {max_title_pages} 頁。這一步只是 Python 的 dict 查詢，不用 GPU、也不用斷詞。dev 平均每題命中 {avg_title_matches} 頁，完全沒有命中任何標題的有 {no_title_match}。
* **(2) 向量排序**：候選頁用 `BAAI/bge-m3`（{retriever_params} 參數）把「頁名：開頭 {page_lead_chars} 字」編成向量，claim 也編成向量，兩邊都做 L2 normalization，所以內積就是 cosine similarity。bge-m3 是專門為檢索訓練過的模型，訓練時用的是 [CLS] 位置的向量，所以我照它的設計取 [CLS]，而不是 sample code 的 mean pooling。為了照顧「claim 裡沒有直接寫出頁名」的題目，我另外在候選池（train 標註頁 {n_train_gold_pages} 頁 + 所有 claim 的標題命中頁，合計 {pool_size} 頁）裡用向量再補 {dense_candidates} 頁。
* **(3) 混合分數**：最終分數 = cosine + bonus × [這頁是否被標題命中]。bonus 我在 dev 上從 {bonus_grid} 裡挑 Recall@{doc_k} 最高的，結果是 **{best_bonus}**，取前 {doc_k} 頁進入下一階段。

**Dev 結果**：dev 共 {n_dev} 筆，Recall 只計算 SUPPORTS / REFUTES 的 {n_verifiable} 筆；某一組標準證據需要的頁面「全部」都在前 N 頁裡才算命中。

**Document Retrieval Recall@{doc_k} = {doc_r5}（{doc_hits} / {n_verifiable}）**

{doc_table}

{doc_compare_text}另外，「標題命中 + 向量補充」的候選頁涵蓋率是 {candidate_coverage}，這是排序再怎麼改都超不過的上限。

2. Describe your method for retrieving evidence (5%) and report Recall@N on development set  (5%).

Answer:

證據只能從第 1 階段取回的 {doc_k} 頁裡挑。sample code 只把 train 標註過的句子當候選，我改成把這幾頁的**每一句**都當候選（保留維基原始的句號，不重新編號），全部共編碼了 {n_sentences} 句。每一句前面我都加上頁名（「頁名：句子」），因為很多句子是用「他」「該片」來代稱主角，不加頁名的話模型不知道這句在講誰。（這一段 v2、v3 相同。）

* **(1) 向量初篩**：用同一個 bge-m3 算 claim 和每一句的 cosine similarity，每題先留前 {sentence_candidates} 句。v1 只留 20 句，但我發現 reranker 再厲害也只能從初篩留下的句子裡挑，所以 v2 放寬到 {sentence_candidates} 句，提高上限。
* **(2) Cross-encoder 重排**：再用 `BAAI/bge-reranker-v2-m3`（{reranker_params} 參數）把「claim + 句子」一起丟進模型打分數。向量檢索是 bi-encoder，claim 和句子各自算向量再比，速度快但兩邊沒有真正一起讀；cross-encoder 讓 claim 和句子的每個字互相做 attention，比較能分辨同一頁裡內容很接近的句子。
* **(3) 用 CFEVER train 微調 reranker（v2 新增）**：現成的 reranker 只學過一般的「這句和問題相不相關」，但 CFEVER 要的是「能用來判斷真假的那一句」，而且標註者挑句子有自己的習慣。所以我用 train 的 {n_rerank_examples} 題 SUPPORTS / REFUTES 微調它：每題組成 1 個正例（標準證據句）+ {rerank_negatives} 個 hard negative（bge-m3 初篩排很前面、但不是標準證據的句子），reranker 對 8 句各打一個分數，再把 8 個分數當成「8 選 1」做 cross entropy，目標是讓標準證據的分數比其他 7 句都高（listwise loss，參考 FlagEmbedding 訓練 bge-reranker 的方式）。為了讓 T4 放得下，我凍結了詞向量層（約 2.5 億參數），只訓練後面的 transformer 層；learning rate {rerank_lr}、1 個 epoch、fp16 混合精度，大約花 {rerank_minutes} 分鐘。dev 完全沒有拿來訓練，微調完會和現成的模型在 dev 上比較，比較好才採用，最後用的是**{reranker_choice}** reranker。
* **(4) 固定給滿 {ev_k} 句**：FEVER score 只看前 5 句，多給不會扣分，所以每題都輸出 {ev_k} 句，盡量提高命中機會。

**Dev 結果**：同樣只算 {n_verifiable} 筆可驗證的題目，頁名和句號都要對，而且要完整涵蓋某一組標準證據才算命中。

**Evidence Retrieval Recall@{ev_k} = {ev_r5}（{ev_hits} / {n_verifiable}）**

{evidence_table}

{evidence_compare_text}如果只看「文件已經找對」的題目，證據 Recall@{ev_k} 是 {ev_given_doc}：{evidence_bottleneck_text}

3. Describe your classification method (5%) and report Accuracy on development set  (5%).

Answer:

v2 的三階段漏斗顯示，dev 裡「證據已經找齊、標籤卻判錯」的 SUPPORTS / REFUTES 有 417 題，比文件檢索（223 題）和證據檢索（60 題）掉的加起來還多，所以 v3 前兩個階段完全沿用 v2，只改分類。v2 只用一個 `hfl/chinese-roberta-wwm-ext-large`；v3 用同一份訓練資料訓練 {n_models} 個不同的模型，再把它們的機率平均（ensemble）。每個模型都是在預訓練模型上接一個三分類層（SUPPORTS / REFUTES / NOT ENOUGH INFO）：

{classifier_table}

選這三個是因為它們「錯的地方不一樣」，平均起來才有幫助：

* `hfl/chinese-roberta-wwm-ext-large`：v2 用的模型，24 層、hidden size 1024，用中文語料做全詞遮罩（whole word masking）預訓練。
* `hfl/chinese-macbert-large`：大小和詞表都一樣，但預訓練時要猜的字不是換成 [MASK]，而是換成意思相近的詞（MLM as correction），預訓練和 fine-tune 看到的文字比較一致。
* `FacebookAI/xlm-roberta-large`：100 種語言一起預訓練，用 SentencePiece 斷詞，繁體字原生支援。它的詞表有 25 萬個，詞向量就佔了快一半參數，所以我凍結詞向量層（和 v2 微調 reranker 時一樣），省下梯度和 Adam 狀態的記憶體；它的學習率也調得比較小，因為 XLM-R large 用太大的學習率比較容易學不起來。

三個都是只做過語言模型預訓練的模型，沒有用其他標註資料集微調過，符合「只用 CFEVER」的規定。

每個模型的輸入和訓練資料都和 v2 相同：輸入是句對 `claim` + `【頁名】證據句1 【頁名】證據句2 …`，上限 {max_length} token、每句證據最多 {sentence_char_limit} 字、太長時只截斷證據不截斷 claim。SUPPORTS / REFUTES 的訓練資料如果標準證據不在檢索結果裡，就補進一組並放在最前面（{n_gold_injected} 筆有補）；NOT ENOUGH INFO 則用 reranker 從向量前 {nei_candidates} 句挑 5 句，讓它和 test 時一樣「看起來很相關、但其實資訊不夠」。我也檢查過這個格式和 test 是否一致：dev 上證據有找齊的題目中，有 {gold_top1_ratio} 的標準證據被 reranker 排在第 1 句，和訓練時「標準證據放最前面」吻合。

訓練用了 {n_train_used} 筆 train。設定是 AdamW、每批 {batch_size} 筆並用 gradient accumulation 累積 {grad_accum} 批才更新一次（等效 batch size {effective_batch}）、{epochs} 個 epoch、前 {warmup_pct} 的更新步數做 warmup 之後線性下降、gradient clipping 1.0、fp16 混合精度。每個 epoch 結束都在 dev 上看一次，各自採用 dev FEVER score 最高的 epoch：

{epoch_tables}

**Ensemble**：每個模型輸出的 logits 先做 softmax 變成三類機率，取平均後再取 log 當作新的分數（對單一模型來說，log softmax 和原本的 logits 只差一個常數，取 argmax 的結果不變）。我把所有模型組合（單一模型、兩兩一組、三個一起）都在 dev 上比過，每個組合各自挑最好的 NEI bias 和 reranker 門檻（第 4 題說明）：

{ensemble_table}

{ensemble_text}最後用的是 **{chosen_combo}**，NEI bias = {best_bias}、門檻 = {best_threshold}。

**Dev Accuracy = {acc_best}**（dev 全部 {n_dev} 筆，包含 NEI；分類器看的是實際檢索到的證據，這裡只比對標籤）。不做任何調整時是 {acc_plain}。混淆矩陣如下（列是正確答案、欄是預測）：

{confusion_table}

4. … Anything that can strengthen your report. (10%)

Answer:

**(1) v1 → v2 → v3 改了什麼、進步多少**：v1 的 public score 是 {v1_public}（排行榜第 {v1_rank} 名）。v2 保留 v1 的文件檢索，改進證據和分類兩段：初篩 20 → {sentence_candidates} 句、微調 reranker、分類器 base → large、輸入 256 → {max_length} token 並把標準證據放前面、NEI 訓練證據改用 reranker 挑、NEI 決策加上 reranker 門檻，public score 進步到 {v2_public}（第 {v2_rank} 名）。v3 根據 v2 的三階段漏斗只改分類：{n_models} 個模型的 ensemble，並放寬 NEI bias 和門檻的搜尋範圍。三版在 dev 上的比較如下：

{version_table}

{version_compare_text}

**(2) 證據被截斷的問題（v2 修正）**：{trunc_text}

**(3) 三階段漏斗：分數到底卡在哪一步？** 我把 dev 裡 {n_verifiable} 筆 SUPPORTS / REFUTES 一路追下去，看每一步還剩幾題：

{funnel_table}

{funnel_text}{funnel_compare_text}

**(4) NEI 決策：Accuracy 和 FEVER score 的取捨**。v1 只在 NEI 的分數加 bias；v2 再加一個規則：reranker 給第 1 名句子的分數如果低於門檻，代表連最像的句子都不太像證據，直接判 NEI。v2 搜尋的 bias 範圍是 -1.5～1.5，結果選到邊界的 -1.5，所以 v3 把範圍放寬到 {bias_grid}；門檻取 dev 上第 1 名分數的 {quantile_grid} 分位數。下面是 v3 選用的模型組合只調 NEI bias 的結果：

{bias_table}

NEI bias 和 reranker 門檻一起調（每個門檻只列出該門檻下最好的 bias）：

{rule_table}

{rule_text}

**(5) 標題加分的消融實驗**：bonus = 0 等於只用向量排序，bonus 很大等於「標題命中的頁一定排在前面」。

{bonus_table}

{bonus_text}

**(6) 作業規定的檢查**：這份 notebook 一共用了 5 個預訓練模型，每一個都小於 1B：檢索用的 bge-m3（{retriever_params}）、bge-reranker-v2-m3（{reranker_params}），以及分類用的 {classifier_params_text}。助教 11/06 答覆 1B 的限制是逐一適用於每個模型，ensemble 不合併計算。全部在 Kaggle / Colab 的 GPU 上執行，沒有呼叫任何外部推理 API。訓練資料只用 CFEVER 的 train（reranker 和 3 個分類器都是），dev 只用來挑 bonus、NEI bias、reranker 門檻、ensemble 的模型組合、選 epoch 和報告分數，test 只讀 `id` 和 `claim`，沒有用到 test 的答案。

**(7) 限制與之後想做的事**：

* bonus、NEI bias、門檻、epoch 和 ensemble 的模型組合都是在同一份 dev 上挑的，所以 dev 分數會比真實情況稍微樂觀一點。比較嚴謹的做法是把 dev 切成兩半，一半挑參數、另一半報告分數。
* 文件檢索還有 {doc_miss} 的 SUPPORTS / REFUTES 沒有找齊證據頁，常見的是 claim 用的是別名或比較口語的說法（例如 claim 寫「爬蟲類」，頁名是「爬行動物」），v3 沒有處理這部分，之後可以讓頁面也用 cross-encoder 重排。
* reranker 是一句一句分開打分數，需要 2 句以上一起才能判斷的題目，第 2 句單獨看常常不像證據，容易被排到後面。之後可以讓 reranker 一次看多句。
* 參數挑好之後，可以把 train 和 dev 合起來重新訓練最終的分類器，資料多 12%，但就沒有 dev 可以檢查了。

5. Generative AI Usage Statement

Answer:

我有使用生成式 AI，使用情況如下：

* **使用工具**：主要是 Claude（Anthropic，透過 Claude Code）；另外也把題目丟給 ChatGPT 和 Gemini 問過提升分數的方向，再由我比較後決定最後的做法。
* **使用範圍**：(1) 理解 CFEVER 三階段流程和 FEVER score 的計算方式；(2) 協助把 sample code 改寫成「標題比對 + bge-m3 + reranker + RoBERTa」的 v1；v1 拿到 public score 0.776 之後，討論並實作 v2 的改進（微調 reranker、換成 large 分類器、處理證據截斷、NEI 決策規則）；v2 拿到 0.856 之後，依照 v2 的三階段漏斗決定 v3 只改分類，並實作 3 個分類器的 ensemble；還有自動把數字填進報告的程式；(3) 整理學習筆記、練習題和我可能會問的問題；(4) 報告初稿的文字整理。
* **主要 prompt**：「依照作業規範改寫 sample code 並回答 report」、「評估 Claude 和 Gemini 提出的提分建議，哪個在 GPU 時數內比較可行」、「v1 只有 0.776、排第 64 名，還有什麼方法可以提升，幫我寫 v2，一樣要遵守作業規範」、「目前 v2 是 0.85，要不要試試看 v3」、「整理我讀這份程式可能會問的問題」。
* 報告裡的數字都是我這次在 Kaggle / Colab 實際執行後印出來的結果，不是 AI 生成的；程式最上方的註解也有標註生成式 AI 的使用和參考連結。'''
report_text = REPORT_TEMPLATE.format_map(v)
(OUTPUT_DIR / 'report_filled.md').write_text(report_text, encoding='utf-8')

metrics = {key: value for key, value in v.items() if isinstance(value, (int, float, str)) and '_table' not in key}
(OUTPUT_DIR / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(report_text)
```

# Report

Your Kaggle username: zhi816

Your public leaderboard score: （提交 Kaggle 後填入）

Your private leaderboard score: （11/23 公布後填入）


1. Describe your method for retrieving Wikipedia documents (5%) and report Recall@N on development set (5%).

Answer:

我讀 sample code 時發現，它只把 train 有標註過的頁面當候選，dev / test 的正確頁面如果沒在 train 出現過，就一定找不到。所以我改成讀完整的 `wiki_pages.jsonl`（共 【待填】 頁），並把文件檢索拆成「先找候選、再排順序」兩步（這一段 v1、v2、v3 三個版本都相同）：

* **(1) 標題比對（找候選）**：維基的頁名常常就是 claim 裡的主角，例如 claim 在講某部電影，頁名通常就是那部電影的名字。我先把所有頁名正規化（去掉結尾的消歧義括號、統一「·」這類間隔號、去掉空白、英文轉小寫），做成「標題 → 頁面」的 dict，再把 claim 中所有長度 2～30 的子字串拿去查。查到的就是候選頁，長的標題比較具體所以排前面，每題最多留 30 頁。這一步只是 Python 的 dict 查詢，不用 GPU、也不用斷詞。dev 平均每題命中 【待填】 頁，完全沒有命中任何標題的有 【待填】。
* **(2) 向量排序**：候選頁用 `BAAI/bge-m3`（568M 參數）把「頁名：開頭 200 字」編成向量，claim 也編成向量，兩邊都做 L2 normalization，所以內積就是 cosine similarity。bge-m3 是專門為檢索訓練過的模型，訓練時用的是 [CLS] 位置的向量，所以我照它的設計取 [CLS]，而不是 sample code 的 mean pooling。為了照顧「claim 裡沒有直接寫出頁名」的題目，我另外在候選池（train 標註頁 【待填】 頁 + 所有 claim 的標題命中頁，合計 【待填】 頁）裡用向量再補 30 頁。
* **(3) 混合分數**：最終分數 = cosine + bonus × [這頁是否被標題命中]。bonus 我在 dev 上從 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0 裡挑 Recall@5 最高的，結果是 **【待填】**，取前 5 頁進入下一階段。

**Dev 結果**：dev 共 【待填】 筆，Recall 只計算 SUPPORTS / REFUTES 的 【待填】 筆；某一組標準證據需要的頁面「全部」都在前 N 頁裡才算命中。

**Document Retrieval Recall@5 = 【待填】（【待填】 / 【待填】）**

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】另外，「標題命中 + 向量補充」的候選頁涵蓋率是 【待填】，這是排序再怎麼改都超不過的上限。

2. Describe your method for retrieving evidence (5%) and report Recall@N on development set  (5%).

Answer:

證據只能從第 1 階段取回的 5 頁裡挑。sample code 只把 train 標註過的句子當候選，我改成把這幾頁的**每一句**都當候選（保留維基原始的句號，不重新編號），全部共編碼了 【待填】 句。每一句前面我都加上頁名（「頁名：句子」），因為很多句子是用「他」「該片」來代稱主角，不加頁名的話模型不知道這句在講誰。（這一段 v2、v3 相同。）

* **(1) 向量初篩**：用同一個 bge-m3 算 claim 和每一句的 cosine similarity，每題先留前 30 句。v1 只留 20 句，但我發現 reranker 再厲害也只能從初篩留下的句子裡挑，所以 v2 放寬到 30 句，提高上限。
* **(2) Cross-encoder 重排**：再用 `BAAI/bge-reranker-v2-m3`（568M 參數）把「claim + 句子」一起丟進模型打分數。向量檢索是 bi-encoder，claim 和句子各自算向量再比，速度快但兩邊沒有真正一起讀；cross-encoder 讓 claim 和句子的每個字互相做 attention，比較能分辨同一頁裡內容很接近的句子。
* **(3) 用 CFEVER train 微調 reranker（v2 新增）**：現成的 reranker 只學過一般的「這句和問題相不相關」，但 CFEVER 要的是「能用來判斷真假的那一句」，而且標註者挑句子有自己的習慣。所以我用 train 的 【待填】 題 SUPPORTS / REFUTES 微調它：每題組成 1 個正例（標準證據句）+ 7 個 hard negative（bge-m3 初篩排很前面、但不是標準證據的句子），reranker 對 8 句各打一個分數，再把 8 個分數當成「8 選 1」做 cross entropy，目標是讓標準證據的分數比其他 7 句都高（listwise loss，參考 FlagEmbedding 訓練 bge-reranker 的方式）。為了讓 T4 放得下，我凍結了詞向量層（約 2.5 億參數），只訓練後面的 transformer 層；learning rate 2e-05、1 個 epoch、fp16 混合精度，大約花 【待填】 分鐘。dev 完全沒有拿來訓練，微調完會和現成的模型在 dev 上比較，比較好才採用，最後用的是**【待填】** reranker。
* **(4) 固定給滿 5 句**：FEVER score 只看前 5 句，多給不會扣分，所以每題都輸出 5 句，盡量提高命中機會。

**Dev 結果**：同樣只算 【待填】 筆可驗證的題目，頁名和句號都要對，而且要完整涵蓋某一組標準證據才算命中。

**Evidence Retrieval Recall@5 = 【待填】（【待填】 / 【待填】）**

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】如果只看「文件已經找對」的題目，證據 Recall@5 是 【待填】：【待填：第 7 節自動產生的分析句】

3. Describe your classification method (5%) and report Accuracy on development set  (5%).

Answer:

v2 的三階段漏斗顯示，dev 裡「證據已經找齊、標籤卻判錯」的 SUPPORTS / REFUTES 有 417 題，比文件檢索（223 題）和證據檢索（60 題）掉的加起來還多，所以 v3 前兩個階段完全沿用 v2，只改分類。v2 只用一個 `hfl/chinese-roberta-wwm-ext-large`；v3 用同一份訓練資料訓練 3 個不同的模型，再把它們的機率平均（ensemble）。每個模型都是在預訓練模型上接一個三分類層（SUPPORTS / REFUTES / NOT ENOUGH INFO）：

【待填：執行第 7 節後，貼上印出的表格】

選這三個是因為它們「錯的地方不一樣」，平均起來才有幫助：

* `hfl/chinese-roberta-wwm-ext-large`：v2 用的模型，24 層、hidden size 1024，用中文語料做全詞遮罩（whole word masking）預訓練。
* `hfl/chinese-macbert-large`：大小和詞表都一樣，但預訓練時要猜的字不是換成 [MASK]，而是換成意思相近的詞（MLM as correction），預訓練和 fine-tune 看到的文字比較一致。
* `FacebookAI/xlm-roberta-large`：100 種語言一起預訓練，用 SentencePiece 斷詞，繁體字原生支援。它的詞表有 25 萬個，詞向量就佔了快一半參數，所以我凍結詞向量層（和 v2 微調 reranker 時一樣），省下梯度和 Adam 狀態的記憶體；它的學習率也調得比較小，因為 XLM-R large 用太大的學習率比較容易學不起來。

三個都是只做過語言模型預訓練的模型，沒有用其他標註資料集微調過，符合「只用 CFEVER」的規定。

每個模型的輸入和訓練資料都和 v2 相同：輸入是句對 `claim` + `【頁名】證據句1 【頁名】證據句2 …`，上限 512 token、每句證據最多 120 字、太長時只截斷證據不截斷 claim。SUPPORTS / REFUTES 的訓練資料如果標準證據不在檢索結果裡，就補進一組並放在最前面（【待填】 筆有補）；NOT ENOUGH INFO 則用 reranker 從向量前 10 句挑 5 句，讓它和 test 時一樣「看起來很相關、但其實資訊不夠」。我也檢查過這個格式和 test 是否一致：dev 上證據有找齊的題目中，有 【待填】 的標準證據被 reranker 排在第 1 句，和訓練時「標準證據放最前面」吻合。

訓練用了 【待填】 筆 train。設定是 AdamW、每批 8 筆並用 gradient accumulation 累積 2 批才更新一次（等效 batch size 16）、2 個 epoch、前 10% 的更新步數做 warmup 之後線性下降、gradient clipping 1.0、fp16 混合精度。每個 epoch 結束都在 dev 上看一次，各自採用 dev FEVER score 最高的 epoch：

【待填：執行第 7 節後，貼上印出的表格】

**Ensemble**：每個模型輸出的 logits 先做 softmax 變成三類機率，取平均後再取 log 當作新的分數（對單一模型來說，log softmax 和原本的 logits 只差一個常數，取 argmax 的結果不變）。我把所有模型組合（單一模型、兩兩一組、三個一起）都在 dev 上比過，每個組合各自挑最好的 NEI bias 和 reranker 門檻（第 4 題說明）：

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】最後用的是 **【待填】**，NEI bias = 【待填】、門檻 = 【待填】。

**Dev Accuracy = 【待填】**（dev 全部 【待填】 筆，包含 NEI；分類器看的是實際檢索到的證據，這裡只比對標籤）。不做任何調整時是 【待填】。混淆矩陣如下（列是正確答案、欄是預測）：

【待填：執行第 7 節後，貼上印出的表格】

4. … Anything that can strengthen your report. (10%)

Answer:

**(1) v1 → v2 → v3 改了什麼、進步多少**：v1 的 public score 是 0.77600（排行榜第 64 名）。v2 保留 v1 的文件檢索，改進證據和分類兩段：初篩 20 → 30 句、微調 reranker、分類器 base → large、輸入 256 → 512 token 並把標準證據放前面、NEI 訓練證據改用 reranker 挑、NEI 決策加上 reranker 門檻，public score 進步到 0.85600（第 50 名）。v3 根據 v2 的三階段漏斗只改分類：3 個模型的 ensemble，並放寬 NEI bias 和門檻的搜尋範圍。三版在 dev 上的比較如下：

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】

**(2) 證據被截斷的問題（v2 修正）**：【待填：第 7 節自動產生的分析句】

**(3) 三階段漏斗：分數到底卡在哪一步？** 我把 dev 裡 【待填】 筆 SUPPORTS / REFUTES 一路追下去，看每一步還剩幾題：

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】【待填：第 7 節自動產生的分析句】

**(4) NEI 決策：Accuracy 和 FEVER score 的取捨**。v1 只在 NEI 的分數加 bias；v2 再加一個規則：reranker 給第 1 名句子的分數如果低於門檻，代表連最像的句子都不太像證據，直接判 NEI。v2 搜尋的 bias 範圍是 -1.5～1.5，結果選到邊界的 -1.5，所以 v3 把範圍放寬到 -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0；門檻取 dev 上第 1 名分數的 2.5%, 5%, 7.5%, 10%, 12.5%, 15%, 20% 分位數。下面是 v3 選用的模型組合只調 NEI bias 的結果：

【待填：執行第 7 節後，貼上印出的表格】

NEI bias 和 reranker 門檻一起調（每個門檻只列出該門檻下最好的 bias）：

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】

**(5) 標題加分的消融實驗**：bonus = 0 等於只用向量排序，bonus 很大等於「標題命中的頁一定排在前面」。

【待填：執行第 7 節後，貼上印出的表格】

【待填：第 7 節自動產生的分析句】

**(6) 作業規定的檢查**：這份 notebook 一共用了 5 個預訓練模型，每一個都小於 1B：檢索用的 bge-m3（568M）、bge-reranker-v2-m3（568M），以及分類用的 【待填：第 7 節自動產生的分析句】。助教 11/06 答覆 1B 的限制是逐一適用於每個模型，ensemble 不合併計算。全部在 Kaggle / Colab 的 GPU 上執行，沒有呼叫任何外部推理 API。訓練資料只用 CFEVER 的 train（reranker 和 3 個分類器都是），dev 只用來挑 bonus、NEI bias、reranker 門檻、ensemble 的模型組合、選 epoch 和報告分數，test 只讀 `id` 和 `claim`，沒有用到 test 的答案。

**(7) 限制與之後想做的事**：

* bonus、NEI bias、門檻、epoch 和 ensemble 的模型組合都是在同一份 dev 上挑的，所以 dev 分數會比真實情況稍微樂觀一點。比較嚴謹的做法是把 dev 切成兩半，一半挑參數、另一半報告分數。
* 文件檢索還有 【待填】 的 SUPPORTS / REFUTES 沒有找齊證據頁，常見的是 claim 用的是別名或比較口語的說法（例如 claim 寫「爬蟲類」，頁名是「爬行動物」），v3 沒有處理這部分，之後可以讓頁面也用 cross-encoder 重排。
* reranker 是一句一句分開打分數，需要 2 句以上一起才能判斷的題目，第 2 句單獨看常常不像證據，容易被排到後面。之後可以讓 reranker 一次看多句。
* 參數挑好之後，可以把 train 和 dev 合起來重新訓練最終的分類器，資料多 12%，但就沒有 dev 可以檢查了。

5. Generative AI Usage Statement

Answer:

我有使用生成式 AI，使用情況如下：

* **使用工具**：主要是 Claude（Anthropic，透過 Claude Code）；另外也把題目丟給 ChatGPT 和 Gemini 問過提升分數的方向，再由我比較後決定最後的做法。
* **使用範圍**：(1) 理解 CFEVER 三階段流程和 FEVER score 的計算方式；(2) 協助把 sample code 改寫成「標題比對 + bge-m3 + reranker + RoBERTa」的 v1；v1 拿到 public score 0.776 之後，討論並實作 v2 的改進（微調 reranker、換成 large 分類器、處理證據截斷、NEI 決策規則）；v2 拿到 0.856 之後，依照 v2 的三階段漏斗決定 v3 只改分類，並實作 3 個分類器的 ensemble；還有自動把數字填進報告的程式；(3) 整理學習筆記、練習題和我可能會問的問題；(4) 報告初稿的文字整理。
* **主要 prompt**：「依照作業規範改寫 sample code 並回答 report」、「評估 Claude 和 Gemini 提出的提分建議，哪個在 GPU 時數內比較可行」、「v1 只有 0.776、排第 64 名，還有什麼方法可以提升，幫我寫 v2，一樣要遵守作業規範」、「目前 v2 是 0.85，要不要試試看 v3」、「整理我讀這份程式可能會問的問題」。
* 報告裡的數字都是我這次在 Kaggle / Colab 實際執行後印出來的結果，不是 AI 生成的；程式最上方的註解也有標註生成式 AI 的使用和參考連結。
