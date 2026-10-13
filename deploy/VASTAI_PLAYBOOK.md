# VASTAI_PLAYBOOK.md — read before touching vast.ai

> Source: lessons from a previous GPU-automation task that wasted real money
> (two idle machines, ~$0.5 burned with zero output) plus this project's own
> experience. **Money and safety rules here win over any other doc.**
> ⛔ Read this and `AGENTS.md` before renting anything.

---

## 0. One-line root cause

> **Prove it is alive, translate one sentence, only then run a batch; judge by
> content not line counts; write remote commands to files; always verify teardown.**

Most of the waste comes from **running unvalidated install/setup scripts on a
paid machine for the first time**. Bake the environment into an image and test it
at build time (see `deploy/Dockerfile`), and this whole class of failure moves to
a free step.

---

## 1. Root causes (four classes)

| Class | Symptom | Countermeasure |
|---|---|---|
| **A. Failures are hidden** | `cmd \| tail` always exits 0; a dead server keeps appending empty output; the wrong health endpoint 404s forever | Judge success by **content and marker files**, not exit codes |
| **B. Environment assumed, not verified** | binary needs a newer glibc; `--chat_format` value is invalid; a CLI entry point is not on PATH; LD path order hides a newer libstdc++ | Verify external deps **locally first**; single-sentence test before a batch |
| **C. Wrong order** | hundreds of batches launched before any single test → all empty | **Three gates**: alive → single sentence → no stale process |
| **D. Tooling friction** | inline ssh quoting mangled; background process keeps the ssh session's fds | Write remote commands to `.sh`, scp, then run; redirect all three fds |

Plus two human errors:

| Class | Symptom | Countermeasure |
|---|---|---|
| **E. Fixing a bad machine** | slow network but keep switching proxies / retrying setup | **If the network is bad, switch machines** — don't debug the host |
| **F. Stop without revoking** | machine keeps running after "stop" | Stop = destroy **and** revoke keys |

---

## 2. Pitfall table

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| V01 | non-ASCII `print` raises `UnicodeEncodeError` on Windows | console codepage | `sys.stdout.reconfigure(utf-8)`; `chcp 65001` |
| V02 | line counts double after writing files | Windows text mode writes `\r\n` | write with `newline="\n"` |
| V03 | `~/.ssh` not writable | external dir permissions | keep keys in the workspace; delete after |
| V04 | `ssh-keygen -N ""` fails on Windows | empty passphrase + interactive prompt | generate the key in Python (`cryptography`) |
| V05 | instance-create output fails to parse | it is `"Started. {...}"`, not pure JSON | regex the braces; retry `show instance` |
| V06 | `Permission denied (publickey)` | **private key ACL too open → ssh ignores the key**; also key not yet propagated | `icacls <key> /inheritance:r /grant:r "%USERNAME%:F"`; wait for ssh to answer |
| V07 | pulled Q4 instead of best quality | the 14B repo has no file literally named `Q6_K` (no suffix = Q6_K) | trust the **file size**, not the name |
| V08 | remote download 403 / `KeyError` | datacenter IPs are rate-limited by some hosts | **do not call APIs from the remote**; precompute names/URLs locally |
| V09 | setup "succeeded" but did nothing | `cmd \| tail` exit code is always 0 | write a log; echo a status marker; verify it |
| V10 | offer query returns nothing for a VRAM filter | that field is not queryable in offers | drop it; 24GB cards are known |
| V11 | translation line count mismatch | model merges/splits newlines | retry, leave blank, verify gaps, continue |
| V12 | bill keeps growing | per-second billing during debugging | set a max price; auto-destroy; **report money first** |
| V13 | `pip install vast-ai` not found | the package is `vastai` | `pip install vastai` |
| V14 | PowerShell lacks `head`/`file`, quoting hell | limited tools | write a file and run it; don't inline |
| V15 | llama.cpp binary won't start | needs a newer glibc than the base image | use a compatible image, or `ollama` (see V39) |
| V16 | unpacking a cudart archive has no binary | it contains only `.so` | pair it with the real binary archive |
| V17 | `huggingface-cli` not found | entry point not on PATH in non-interactive ssh | use `python -c "from huggingface_hub import hf_hub_download"` |
| V18 | package installed but lib not found | non-interactive ssh has no LD path | set `LD_LIBRARY_PATH`, correct order |
| V19 | `/health` always 404 → server wrongly declared dead | that server build has no `/health` | probe `/v1/models` |
| V20 | every request 500, empty output | invalid chat template name | use the right one; single-sentence test first |
| V21 | output file full of empty lines | a dead server appended blanks | delete old output before starting; count non-empty |
| V22 | remote command syntax error | the outer tool ate the quotes | write remote commands to `.sh` |
| V23 | `ssh 'nohup cmd &'` hangs locally | remote background process holds the ssh fds | `nohup cmd > log 2>&1 < /dev/null &` |
| V24 | GB-size file over proxy scp is unusable | relay throttling | download big files **on the remote**; scp only small files |
| V25 | `destroy` says `Aborted`, machine stays up | CLI prompts `[y/N]` in non-interactive mode | `echo y \| vastai destroy instance <id>`; verify |
| V26 | `Invalid user key` | key pasted wrong | vast keys are **64 chars**; check the length |
| V27 | GPU 0%, no model, still billing | idle | destroy after 15 minutes of idle |
| V28 | several `pip install` overlapping | new command sent before the previous finished | `pgrep -af` before sending a new one |
| V29 | network slow/erratic | bad host | network check within 5 minutes; switch if bad |
| V30 | concluding "the site blocks datacenter IPs" from one 403 | single error treated as final | retest 3×, try another URL; cite reproducible evidence |
| V31 | artifacts lost after destroy | nothing pulled back first | **pull and verify before destroying** |
| V32 | cross-platform tool reports all-pass/all-fail | `select()` on a pipe is unsupported on Windows; init/system events also contain the skill name | validate on Windows first; judge only real tool calls/content |
| V33 | structured-extraction task returns schema keys | a translation model is not a general instruction model | don't use it for JSON tasks |
| V34 | `pip install comic-text-detector` not found | it is **not on PyPI** | install from GitHub, or use a fallback detector |
| V35 | OCR model reports "PyTorch not found" though torch is installed | `transformers` 5.x requires torch ≥ 2.5; the image had 2.4 | pin `transformers==4.46.3` |
| V36 | OCR endpoint 500 on first call | `python-multipart` missing | install it |
| V37 | a CLI flag silently ignored | the option is defined both globally and per-subcommand | define it once |
| V38 | output image size/quality drops | PIL's JPEG default is `quality=75` + 4:2:0 chroma subsampling | save with the source quality and `subsampling=0` |
| V39 | LLM server is painful to stand up | binary/glibc and chat-template issues | use **ollama** (OpenAI-compatible, handles the model template) |
| V40 | a stalled model pull looks "ready" | the LLM server answers 200 with an empty model list | readiness requires a **non-empty** model list |
| V41 | a bad-network machine wastes 30 minutes pulling | no speed gate | probe the download speed at boot and fail fast below a threshold |

---

## 3. Hard rules

**R1 Before renting**
1. **No renting without explicit user consent**; consent is per-machine.
2. **Report the budget first**: "N machines × $X/hr × Y hours ≈ $Z".
3. **Write and syntax-check scripts locally** (`bash -n`) before renting; do not
   iterate on a paid machine.
4. Rent only network-qualified offers: reasonable `inet_down`/`inet_up`,
   `reliability` ≥ 0.97, and use `--direct`.

**R2 Within 5 minutes of boot**
5. Run `deploy/net_check.sh` and time a 10MB scp.
6. `NET_BAD`, or 10MB scp > 30s, or only a proxy ssh for a large-file task →
   **destroy and switch**. Do not change proxies, do not fall back to CPU, do not
   retry setup.
7. Three consecutive bad machines → **stop and report**; do not rent a fourth.

**R3 While running**
8. Three gates before a batch: alive (`/v1/models` has a model) → single-sentence
   test 200 → `pgrep -af` shows no stale process.
9. Judge success by **content and marker files**, not line counts or exit codes.
10. `pgrep -af` before sending a new command.
11. GPU idle for 15 minutes with no download → treat as wasted, destroy.
12. Same error fixed twice without success → **stop and report**.
13. Conclusions need a **reproducible test**.

**R4 Budget**
14. **$1 cap per task** (measured on `show user`); report at $0.8, stop at $1.
15. **First line of every report is money.**

**R5 Teardown**
16. On completion, stop, interruption, handoff, or leaving the conversation:
    **destroy first**, confirm `show instances` is empty, confirm billing stopped.
17. `echo y | vastai destroy instance <id>`, then verify.
18. **Pull and verify artifacts before destroying.**
19. Never leave a machine running unattended.

**R6 Access**
20. An agent forbidden from touching vast must not rent, ssh, or destroy.
21. **API keys live only in environment variables; never write them to files.**

## Violations

If you break a rule: **stop, destroy everything, confirm billing stopped, and
report which rule and how much it cost.** Do not hide it.

---

## 4. Network check SOP

Tool: `deploy/net_check.sh` (last line `NET_OK` / `NET_BAD`).

Only the sources actually needed are **required**: PyPI (pip) and HuggingFace
(models). Other sites (e.g. GitHub, which often rate-limits datacenter IPs) are
**informational only** and must not fail the check.

Also time a 10MB scp from your machine: **> 30s → switch**.

**Reading results:** `302` is a normal redirect; `000` is a timeout; a single
`403` is not a conclusion; tens of KB/s means a bad host; `cmd | tail` proves
nothing.

---

## 5. Project-specific notes

| Item | Practice |
|---|---|
| Before a run | `python hanhua.py ping --profile sakura`; `auto --ocr mock` dry run |
| Liveness | probe `/v1/models` (not `/health`) |
| Endpoints | LLM on 8000, OCR on 8080 |
| Model choice | 14B for quality; 7B when VRAM is tight; **trust file size over the GGUF name** |
| Consistency | `terms → review glossary → translate --glossary → normalize → verify` |
| Artifacts | the deliverable is an **archive**; pull it back and verify page count/order before destroying |
| Remote commands | write `.sh` files; **do not call APIs from the remote** |
| Big files | **never scp model weights**; download on the remote (`ollama pull` / `pip install`) |

---

## 6. Checklist

**Before**
- [ ] User consented and the budget was reported
- [ ] Machine selected (reliability floor, `--direct`, max price)
- [ ] Scripts syntax-checked locally
- [ ] Upload list is KB-size scripts only; weights download on the remote
- [ ] Network check passes within 5 minutes
- [ ] Three gates pass

**After**
- [ ] Artifacts pulled back and verified
- [ ] `echo y | vastai destroy instance <id>`
- [ ] `show instances` empty; billing stopped
- [ ] Key removed from the environment
- [ ] Report starts with money
