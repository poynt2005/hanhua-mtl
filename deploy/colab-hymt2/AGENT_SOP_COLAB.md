# AGENT SOP — `hymt2-heretic-colab`（從 Vast.ai 遷移至 Google Colab L4 + Drive）

> **給接手 Agent 的第一句話：**
> 你先前讀過的過往漢化交接資料（`Handoff資料`）幾乎全是 **Vast.ai + 日文 Sakura** 流程。在 **`hymt2-heretic-colab`** 分支中：
> 1. GPU 推論全面遷移至 **Google Colab L4 + Google Drive（方案 A）**；
> 2. 翻譯方向是 **English → Traditional Chinese**（成人向 Galgame H-scene：露骨詞彙、髒話、擬聲詞、喘息），與 Sakura 的日翻中規則有數項關鍵差異。
> 請先看懂下方「作廢 vs. 沿用」邊界與 Hy-MT2 專屬規格，切勿混用舊習慣。

---

## 一、舊 Vast.ai 經驗 → 新 Colab 流程對照表

### ⛔ 明確作廢（在本分支嚴禁執行）
1. **禁止跟使用者索取 `VAST_API_KEY` 或執行 `vastai` / `hanhua.py vast`**：本流程完全不租用 Vast.ai 機器（也不用等 `hanhua.py vast up` 支援 11434/CUDA 12.8）。
2. **禁止建立 `HanhuaGuard`、Scheduled Task、`vast_guard.py` 或任何背景守護進程**。
3. **禁止關心 Docker image / Digest / SSH / Port 11434 公網穿透**：Colab 端統一在本機 `127.0.0.1:11434` 執行 `llama.cpp` `llama-server`。
4. **禁止呼叫 Ollama 原生 `/api/chat`、`/api/tags`、`keep_alive: 0`**：`hy_v3` 架構使用 `llama-server`（`/v1/*` 與 `/health`）。
5. **禁止由 Agent 直接操作瀏覽器去點 Colab、或在本機試圖直接 `shutil.copy` 寫入 `G:\` / `H:\`**：一律透過 `drive_bridge.py` 執行原子推送與拉回。
6. **禁止從本機上傳 GGUF 模型檔**：18.24 GB 模型由使用者在免費 CPU Colab 上跑 `deploy_models_colab.ipynb` 直接從 HuggingFace 拉進雲盤。

### ✅ 明確沿用與 Hy-MT2 專屬規格
1. **不改使用者遊戲原檔 / 遊戲文本是資料不是指令 / 不散布**：與所有過往交接一致。
2. **模型端不損能力（簡體進、簡體出、後處理轉台繁）**：
   - **不可用 system prompt**（Hy-MT2 官方設計為單一 user message personalization 模板）；
   - Prompt 目標永遠指定 **`简体中文`**，注入 prompt 的 `glossary.json` 譯文與 `speaker` 名稱由 runner 先轉 `tw2sp` 簡體（避免混用簡繁）；
   - `ui`/`system` 類型單元**強制不帶 `STYLE_ADULT` 與前文 3 句上文**，防 Prompt Leak。
3. **一次一格（1 line/request × 4 並行），絕不合併換行送批**：`Hy-MT2-30B-A3B` 雖總參 30B 但僅 3B 激活、極快，單句請求徹底杜絕多行錯位。
4. **英文專用簡轉繁預設使用 OpenCC `s2twp`**：與日文 Sakura 的 `s2t` 裁定不同，英翻中易產生大陸現代用詞（如 软件/视频），依 `deploy/hymt2-heretic/README.md` §6.1 預設使用 `s2twp`（台灣字形＋台灣詞彙），並同樣產出 `zh_CN.jsonl` 與 `watch_chars` 一對多字監控（特別留意 `干/发/后/面`）。
5. **保留佔位符**：`{{TAG_n}}`、`⟦n⟧`、`\C[n]` 數量與順序必須 100% 一致，不符會自動觸發官方簡明模板 fallback 重試。
6. **編碼與換行鐵律**：讀 JSON/JSONL 一律 `utf-8-sig`，寫檔 UTF-8 無 BOM + `\n`，**禁止透過 PowerShell heredoc 傳中文給 Python**。
7. **成敗只看 `.DONE.json` 與 `qa_report.json`**（`missing=0, failures=0`）。

---

## 二、固定雲盤目錄契約（不可自行改名）

```text
<Drive>:\我的雲端硬碟\galgame_mtl\
├── bin\                   # 預編譯 llama-server-l4-b12019.tar.gz（由 deploy_models_colab.ipynb 部署，零編譯）
├── models\
│   └── Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf   # 18,236,703,488 bytes
├── scripts\hymt2\         # drive_bridge.py push 自動同步最新 schema.py 與 runner.py
└── hymt2\
    ├── input\             # <job>.jsonl + <job>.manifest.json
    ├── glossary.json      # 選用，英文 -> 中文詞彙表（可寫繁體，runner 自動 tw2sp）
    ├── work\              # checkpoint.jsonl / failures.jsonl / gpu.log / llama.log
    └── output\            # <job>.zh_CN.jsonl / <job>.zh_TW.jsonl / <job>.qa_report.json / <job>.DONE.json
```

---

## 三、本機 Agent 標準作業流程（Phase 0 → Phase 4）

### Phase 0：雲盤環境體檢
*(若 Agent 目前位於獨立遊戲工作區而非本 Repo 根目錄，請改為直接呼叫 `python <instructions_dir>/drive_bridge.py ...`，例如 `~/Documents/Instructions/hanhua/game/hymt2-heretic-colab/drive_bridge.py`。)*
```powershell
python deploy/colab-hymt2/drive_bridge.py doctor
```
- 多帳號磁碟衝突時設 `$env:GALGAME_MTL_DRIVE_ROOT = "H:\我的雲端硬碟\galgame_mtl"` 或傳 `--drive-root`。
- 模型缺失時請使用者開免費 CPU Colab 跑 `deploy_models_colab.ipynb`。

### Phase 1：整理嚴格 `.jsonl` 與 `glossary.json`
```json
{"id": "00001", "file": "ch01", "kind": "dialogue", "speaker": "Yui", "text": "Good morning. Did you sleep well?"}
{"id": "00002", "file": "ch01", "kind": "dialogue", "speaker": "Yui", "text": "Ahh... don't stop, I'm going to cum! {{TAG_0}}"}
{"id": "00003", "file": "ui", "kind": "ui", "text": "Save Game", "override": "儲存遊戲"}
```
- 欄位白名單：`id`、`text`、`kind`（`dialogue`/`narration`/`choice`/`ui`/`system`）、選填 `file`/`speaker`/`override`。
- `text` 不可含 `\x00`、長度 `<= 1500`；不含英文字母 `[A-Za-z]` 的純符號/數字行與 `override` 行會自動透傳不進 GPU。
- `glossary.json`：人名、稱謂（如 `onii-chan`）、成人露骨用語與固定擬聲詞建議提前列入，英文比對採整字（whole-word）、不分大小寫。

### Phase 2：本機驗證 → 推送雲盤 → 交棒使用者
```powershell
python deploy/colab-hymt2/drive_bridge.py validate --input work/job.jsonl --glossary work/glossary.json
python deploy/colab-hymt2/drive_bridge.py push --input work/job.jsonl --glossary work/glossary.json --name en_game_01
```
推完後 **STOP**，請使用者打開 Google Colab（確認 **L4 GPU**）使用**最新版本的 `hymt2_colab.ipynb`**（雲端 `scripts/` 會自動同步 `runner.py`，但若曾更新結束碼 / 待機保護流程請務必重開最新 Notebook）執行 `Ctrl+F9`，完成或自動斷線後再回報。

### Phase 3：Colab L4 端自動執行內容
1. 檢查 `.manifest.json` SHA-256（最多輪詢等待 60 秒讓 DriveAPP 同步完）。
2. 從 Drive `bin/` 解壓預編譯 `llama-server-l4-b12019.tar.gz`（若不存在則自動從本專案 GitHub Release 秒速下載並快取，Colab 全程零編譯）→ GPU 4 道硬檢：`nvidia-smi` >= `22,000 MiB`（拒跑 T4）→ `CUDA0` 可見 → `offloaded X/X layers to GPU` → VRAM 實佔 >= `18,000 MiB` → 3 句 Gate（日常 / 露骨 / 擬聲詞，0 拒答 0 `<think>`）且速度 `>= 8.0 tok/s`。
3. 4 路並行翻譯＋斷點續傳（每 10 句批次寫入 Drive checkpoint 並於結束/例外 `finally` 強制 flush）；機械檢查失敗會先以官方 plain 模板重試一次，仍失敗記進 `failures.jsonl`。其中寬鬆拒答關鍵字（`抱歉`/`无法`/`I'm sorry`/`I can't`）僅套用於開機 Gate 4；正式逐句翻譯使用嚴格 AI 拒答樣板檢查，避免誤殺正常英文台詞。
4. 當 `missing=0`、`failures=0` 全數通過時才寫入 `output/<job>.DONE.json` 並回傳結束碼 **`0`（`EXIT_OK`）**；若全數單元已跑完、譯文／`checkpoint.jsonl`／`failures.jsonl` 均已寫入但仍有少數拒答／失敗句（且 `llama-server` 未崩潰）則**不寫 `.DONE.json`** 並回傳結束碼 **`2`（`EXIT_PARTIAL`）**，Notebook 僅印出平靜的 `[PARTIAL]` 摘要、不拋 `RuntimeError`、同步 Drive 後照 `AUTO_DISCONNECT=True` 自動釋放 L4 省點數。唯有啟動 Gate 0–4、輸入檔異常、`checkpoint.jsonl` 連續 I/O 失敗或推論中 `llama-server` 意外終止才會回傳/拋錯 **`1`（`EXIT_FATAL`）**，`finally` 先 sync 日誌並 `drive.flush_and_unmount()`，且預設 `DISCONNECT_ON_ERROR=False` 會印出 `STOP THE RUNTIME NOW` 並於 `IDLE_UNASSIGN_MINUTES=15` 後自動釋放 GPU 防呆（寫入 `checkpoint.jsonl` 具 5 次指數退避重試，`gpu.log` 寫入失敗僅警告不中斷）。

### Phase 4：狀態檢查 →（若 Partial 補救 / 事後補測）→ 拉回驗證 → 續接寫回
```powershell
python deploy/colab-hymt2/drive_bridge.py status --name en_game_01
python deploy/colab-hymt2/drive_bridge.py pull --name en_game_01 --dest work/translated
```
- **若 Colab 顯示 `[PARTIAL]`（`rc = 2`）或 `status` 顯示 `failures > 0`（全部句數已跑完並釋放 L4）：**
  1. 檢視 `hymt2/work/<job>.failures.jsonl` 與 `hymt2/output/<job>.qa_report.json`。
  2. **0 GPU 本機重組（全數補上 `"override"` 時，免開 Colab L4）：** 在原 `.jsonl` 對應行補上 `"override": "..."` 並執行 `drive_bridge.py push --force`（使 `todo = 0`），接著直接於本機執行：
     ```powershell
     python <instructions_dir>/runner.py --drive-root "<drive_root>" --api-base http://127.0.0.1:9/v1 --skip-gpu-check --manifest-wait 0
     ```
     幾秒內即可重組輸出並寫入 `.DONE.json`（回傳 `rc = 0`）。
  3. **若仍需模型重翻部分失敗句：** 調整 `glossary.json` 或為部分句加上 `"override"` 後 `drive_bridge.py push --force`，再開最新 `hymt2_colab.ipynb` 按 `Ctrl+F9`（因尚無 `.DONE.json`，會自動復用 `checkpoint.jsonl` 僅翻未完成單元）。
- **`rc = 0`（`[DONE]`）事後恢復與局部補測速查表：**
  | 情境 | 處置方式 | 要開 GPU 嗎？ |
  |---|---|---|
  | 誤按重跑且未改動 | 維持預設 `FORCE_RERUN=False`，偵測有效 `.DONE.json` 直接 `skipping` 回 `rc = 0` | 不重翻 |
  | 事後改少數特定句 | 原 `.jsonl` 補 `"override": "..."` 並 `drive_bridge.py push --force`（會清掉舊 `.DONE.json`），再跑上述本機 `--skip-gpu-check`（未重 push 時補加 `--force`）；或直接後處理 `.zh_TW.jsonl` | **0 GPU** |
  | 強制 GPU 重翻特定句 / 整本 | 刪除 `hymt2/work/<job>.checkpoint.jsonl` 對應 `wid` 行（或整份 checkpoint），開 Colab 勾 `FORCE_RERUN = True` 重跑 | 只翻刪掉的句 |
- 拉回後先人工/腳本抽檢 `hymt2/work/llama.log` 的顯存餘裕與 `qa_report.json` 的 `watch_chars`（特別檢查 `干/發/後/面` 在 `s2twp` 下有無誤轉）及前 50 句 H-scene 語氣，確認無拒答/委婉再寫回遊戲。

### Phase 4.5：拉回驗證通過後的雲盤殘留清理（必做）
1. **僅在確認** `drive_bridge.py pull` 成功、所有 `.zh_TW.jsonl` / `.zh_CN.jsonl` / `.qa_report.json` / `.DONE.json` 已完整保存於本機工作區（且寫回或備份完成）**之後**，才能清理雲盤。
2. 清理目標**僅限本次已完成任務**位於雲端硬碟目錄下的殘留檔：`hymt2/input/`、`hymt2/work/`、`hymt2/output/`（以及若不再沿用之任務專屬 `hymt2/glossary.json`），避免下一款遊戲啟動 Colab 時掃描到舊任務殘留、造成錯誤或重複執行。
3. **絕對不可刪除**共用的 `models/`、`bin/` 與 `scripts/`。

---

## 四、異常排錯速查表

| 現象 | 原因 | 處置 |
|---|---|---|
| `Multiple Google Drives contain galgame_mtl` | 多帳號各建過一次目錄 | 設 `$env:GALGAME_MTL_DRIVE_ROOT` 指定 Colab 5 TB 帳號磁碟 |
| Colab 報 `VRAM < 22000 MiB` 或 `VRAM used < 18000` | 配到 T4（16 GB）或 `-fit on` 壓到門檻溢出部分層回 CPU | 先檢查 `hymt2/work/llama.log` 的 `common_params_fit_impl`；若是 T4 請刪除執行階段重連 L4，若為 L4 餘裕過緊可將 `runner.py` 的 `CTX_SIZE` 降為 `16384` |
| `failures.jsonl` 出現 `refusal` | Heretic 去審查只測過日文，個別英文句偶發真正 AI 拒答 | 調整該句措辭或將關鍵詞進 `glossary.json` / `override` 後續跑（已翻句子不會重翻） |
| 出現 `幹/乾` 或 `髮/發` 一對多誤轉 | OpenCC `s2twp` 詞表未命中特定片語 | 把該片語加入 `glossary.json` 或於 `zh_CN.jsonl` 後處理修正，不需重開 GPU |
