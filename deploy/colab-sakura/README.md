# `sakura-ollama-colab` — Google Colab L4 + Drive 一鍵日翻中方案

這條分支將原本 `sakura-ollama` 的日文 Galgame 翻譯端點改造為 **Google Colab L4 (24 GB) + Google Drive (5 TB) 一鍵自動化管線**，不碰 Docker、不租 vast.ai、跑完自動 `flush_and_unmount()` 並 `runtime.unassign()` 斷開停止計費。

---

## 第一次設定（約 5 分鐘，只做一次）

### 1. 本機 Google Drive 電腦版掛載 Colab 帳號（若 Colab 在另一個 Google 帳號）
1. 點 Windows 右下角系統匣的 **Google Drive 圖示** → 右上角頭像 → **「新增另一個帳戶」**。
2. 登入你開 Colab Pro / 5 TB 空間的那個 Google 帳號。
3. 完成後電腦會多出第二顆磁碟（例如原本是 `G:\`，新帳號會是 `H:\我的雲端硬碟`）。
4. 確認該帳號偏好設定為 **「串流處理檔案 (Stream files)」**，才不會吃掉 C 槽空間。

### 2. 在免費 CPU Colab 部署 GGUF 模型與預編譯執行檔（0 GPU 額度）
1. 用 Colab 帳號打開 Google Colab，匯入 `deploy/colab-sakura/deploy_models_colab.ipynb`。
2. 確認右上角執行階段為 **CPU（免費）**，按 **`Ctrl+F9` 全部執行**。
3. 腳本會自動建立 `MyDrive/galgame_mtl/...` 目錄、拉取 `Sakura-Galtransl-14B-v3.8.gguf`（12.12 GB），並將已封裝好的預編譯 L4 `llama-server-l4-b12019.tar.gz` 直接通過雙重 SHA-256 校驗部署進 `MyDrive/galgame_mtl/bin/`（亦可同時勾選 `DEPLOY_HYMT2` 一併部署英文模型）。

---

## 日常一鍵翻譯流程

1. **本機 Agent 整理並推送任務：**
   ```powershell
   python deploy/colab-sakura/drive_bridge.py doctor
   python deploy/colab-sakura/drive_bridge.py push --input work/game.jsonl --glossary work/glossary.json --name game_01
   ```
2. **你打開 `sakura_colab.ipynb`（確認選 L4 GPU）→ 按 `Ctrl+F9`：**
   - **Notebook 重要參數：** `AUTO_DISCONNECT = True`（成功即自動釋放 GPU）；`DISCONNECT_ON_ERROR = False`（預設錯後先同步日誌並印出 `STOP THE RUNTIME NOW` 方便除錯，並以 `IDLE_UNASSIGN_MINUTES = 15` 倒數自動斷線防止持續計費）；僅需補翻剩餘失敗句時可勾選 `REPAIR_ONLY = True` 並填 `TARGET_JOB`。
   - **Colab 端完全不含任何編譯流程**，直接完成：從 Drive `bin/` 解壓預編譯 `llama-server-l4-b12019.tar.gz`（若缺失則從 GitHub Release 秒速下載並驗證 SHA-256 後寫回 Drive `bin/`）→ 模型複製到本地 SSD → **GPU 4 道硬鎖檢查**（拒跑 T4、驗證 100% 層數進 GPU、VRAM >= 12 GB、3 句 Gate >= 8 tok/s）→ 4 路並行 batch 8 翻譯＋SHA-1 `unit_key` 斷點續傳（含引號/佔位符/假名術語表/換行拆段修復、`<job>.overrides.jsonl` 透傳與 `no_cjk`/`latin_only` 語言 QA）→ `s2t` 簡轉繁與 QA → 全數通過始寫入 `.DONE.json` → `flush_and_unmount()` → 自動斷線釋放 L4。
3. **本機 Agent 驗收並拉回結果：**
   ```powershell
   python deploy/colab-sakura/drive_bridge.py status --name game_01
   python deploy/colab-sakura/drive_bridge.py pull --name game_01 --dest work/translated
   ```
   *(若仍有個別專有名詞/殘餘失敗，檢視 `sakura/work/<job>.failures.jsonl` 並放入 `sakura/input/<job>.overrides.jsonl`，用 `REPAIR_ONLY = True` 只補翻該部分即可，已完成單元 100% 不重複翻。)*

---

## 檔案總覽

| 檔案 | 用途 |
|---|---|
| `deploy_models_colab.ipynb` | 免費 CPU Colab 一鍵部署 GGUF 權重與固定目錄結構 |
| `sakura_colab.ipynb` | L4 GPU 一鍵翻譯＋自動存檔同步＋自動斷線 Notebook |
| `schema.py` | 嚴格 JSONL / glossary / manifest / DONE 驗證器（純 stdlib） |
| `runner.py` | Colab 端推論啟動、GPU 4 道驗活、4 路並行翻譯、QA 主程式 |
| `drive_bridge.py` | 本機 Windows Agent 專用雲盤橋接 CLI（`doctor`/`validate`/`push`/`status`/`pull`） |
| `AGENT_SOP_COLAB.md` | 給已讀過 `vast.ai` 過往範例的 Agent 看的完整遷移 SOP |
| `skill/SKILL.md` | 可攜式 Agent Skill 定義 |
| `test_all.py` | 完整離線單元與整合測試（15 項，含嚴格 Prompt Mock SSE、引號/佔位符/多行修復、語言 QA 與 `--repair-only`） |
