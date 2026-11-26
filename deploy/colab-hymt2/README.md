# `hymt2-heretic-colab` — Google Colab L4 + Drive 一鍵英翻中（成人向）方案

這條分支將原本 `hymt2-heretic` 的英文成人 Galgame 翻譯端點改造為 **Google Colab L4 (24 GB) + Google Drive (5 TB) 一鍵自動化管線**，不碰 Docker、不租 vast.ai、不用等 `hanhua.py vast up` 改造、跑完自動 `flush_and_unmount()` 並 `runtime.unassign()` 斷開停止計費。

---

## 第一次設定（約 5–10 分鐘，只做一次）

### 1. 本機 Google Drive 電腦版掛載 Colab 帳號（若 Colab 在另一個 Google 帳號）
1. 點 Windows 右下角系統匣的 **Google Drive 圖示** → 右上角頭像 → **「新增另一個帳戶」**。
2. 登入你開 Colab Pro / 5 TB 空間的那個 Google 帳號（會多出一顆如 `H:\我的雲端硬碟`）。
3. 確認該帳號偏好設定為 **「串流處理檔案 (Stream files)」**，18.24 GB 權重不會佔用本機 C 槽空間。

### 2. 在免費 CPU Colab 部署 GGUF 模型與預編譯執行檔（0 GPU 額度）
1. 用 Colab 帳號打開 Google Colab，匯入 `deploy/colab-hymt2/deploy_models_colab.ipynb`。
2. 確認右上角執行階段為 **CPU（免費）**，按 **`Ctrl+F9` 全部執行**。
3. 腳本會自動建立 `MyDrive/galgame_mtl/...` 目錄、拉取 `Hy-MT2-30B-A3B-uncensored-heretic.i1-Q4_K_M.gguf`（18.24 GB），並將已封裝好的預編譯 L4 `llama-server-l4-b12019.tar.gz` 直接通過雙重 SHA-256 校驗部署進 `MyDrive/galgame_mtl/bin/`（亦可同時勾選 `DEPLOY_SAKURA` 一併部署日文模型）。

---

## 日常一鍵翻譯流程

1. **本機 Agent 整理並推送任務：**
   ```powershell
   python deploy/colab-hymt2/drive_bridge.py doctor
   python deploy/colab-hymt2/drive_bridge.py push --input work/en_game.jsonl --glossary work/glossary.json --name en_01
   ```
2. **你打開 `hymt2_colab.ipynb`（確認選 L4 GPU）→ 按 `Ctrl+F9`：**
   - **Colab 端完全不含任何編譯流程**，直接完成：從 Drive `bin/` 解壓預編譯 `llama-server-l4-b12019.tar.gz`（若缺失則從本專案 GitHub Release 秒速下載並驗證 SHA-256 後寫回 Drive `bin/`）→ 模型複製到本地 SSD → **GPU 4 道硬鎖檢查**（拒跑 T4、驗證 100% 層數進 GPU、VRAM >= 18 GB、3 句 Gate >= 8 tok/s）→ 4 路並行單句翻譯（含 `STYLE_ADULT` + `tw2sp` + plain fallback）＋斷點續傳 → `s2twp` 轉台繁與 QA → 寫入 `.DONE.json` → `flush_and_unmount()` → 自動斷線釋放 L4。
3. **本機 Agent 驗收並拉回結果：**
   ```powershell
   python deploy/colab-hymt2/drive_bridge.py status --name en_01
   python deploy/colab-hymt2/drive_bridge.py pull --name en_01 --dest work/translated
   ```

---

## 檔案總覽

| 檔案 | 用途 |
|---|---|
| `deploy_models_colab.ipynb` | 免費 CPU Colab 一鍵部署 GGUF 權重與固定目錄結構 |
| `hymt2_colab.ipynb` | L4 GPU 一鍵翻譯＋自動存檔同步＋自動斷線 Notebook |
| `schema.py` | 嚴格 JSONL / glossary / manifest / DONE 驗證器（純 stdlib） |
| `runner.py` | Colab 端推論啟動、GPU 4 道驗活、4 路並行 Hy-MT2 翻譯、QA 主程式 |
| `drive_bridge.py` | 本機 Windows Agent 專用雲盤橋接 CLI（`doctor`/`validate`/`push`/`status`/`pull`） |
| `AGENT_SOP_COLAB.md` | 給已讀過 `vast.ai` 過往範例的 Agent 看的完整遷移 SOP |
| `skill/SKILL.md` | 可攜式 Agent Skill 定義 |
| `test_all.py` | 完整離線單元與整合測試（含本機 mock SSE server） |
