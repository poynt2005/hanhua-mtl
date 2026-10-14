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
| V42 | a prebuilt image boots but its services never start | the instance runs its own entrypoint (sshd), so the image `CMD` is not executed | pass an `--onstart-cmd` that runs the start script |
| V43 | a **baked** image still failed on the first OCR call with `Remote end closed connection` | `transformers` performs a Hub *freshness* check on every `from_pretrained` even when the weights are cached; on a host that cannot reach HuggingFace it retries until the request fails | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` at runtime, and bake every weight into fixed cache paths (`HF_HOME`, `TORCH_HOME`, `EASYOCR_MODULE_PATH`) |
| V44 | `detector` was `false` on every machine; detection silently fell back to EasyOCR | `pip install git+comic-text-detector` **cannot work** (no `setup.py`/`pyproject.toml`, no tags, not on PyPI) and the failure was hidden by `\|\| echo`; the package also needs `pyclipper`/`shapely`, and uses `pkg_resources` (removed in setuptools ≥ 81) | **vendor** it at a pinned commit, install its deps, patch out `pkg_resources` and the training-only `wandb` import, and assert it loads in the build smoke test |
| V45 | the detector loaded but detection still degraded to EasyOCR | `detect()` unpacked the wrong return value (`mask, groups, _` instead of `mask, mask_refined, block_list`) and the resulting exception was swallowed | use the real return value; keep the broad `except` but only as a fallback |
| V46 | every batch aborted mid-run with `Remote end closed connection` / `WinError 10054`, looking like a server crash | the host's port forward DROPS non-streaming requests: an LLM completion is silent for the whole generation, so the socket looks idle. Measured **4/6 non-streaming vs 6/6 streaming** | send LLM requests with `stream: true` and reassemble the SSE deltas; add bounded retries |
| V47 | the full volume died at ~page 5 although the 20-page pilot passed | the OCR `/prepare` call is also non-streaming (no bytes until it finishes) and had **no retry**; the pilot only survived by luck | retry the OCR request too, and make errors name the endpoint |
| V48 | a **second machine was rented** while the first was still alive | a leftover ad-hoc script read "my instance vanished" as "slow host, try the next offer" and re-rented; it bypassed the blacklist and the leftover cleanup that `vast up` performs | **SINGLE-INSTANCE INVARIANT**: `up` refuses while one of ours is running/loading; `enforce_single()` + `vast_current.txt`; a background watchdog destroys any extra instance we own. Never write ad-hoc rental code |
| V49 | every `create instance` refused, with no reason | vast rejects a create when the whole `args` string exceeds **16384 chars**, and the injected start payload was ~19.7 KB | gzip+base64 the payload (~7.9 KB) and refuse to send anything oversized |
| V50 | a rental round looked like "no machines available" | `create_instance()` discarded vast's stderr, hiding `no_such_ask` / "Invalid args" | surface the refusal reason when the reply cannot be parsed |
| V51 | ~60 minutes of billing on one host that never finished booting | Docker Hub throughput is **per host**: one pulled the 15.5 GB image at 3.6 MB/s (16% in 12 min), another had the layers warm and reached `running` in **187 s** | add a pull watchdog (give up after N seconds), prefer hosts that already hold the image, and blacklist the slow host |
| V52 | offers returned by the query are all refused | vast's bundle list goes **stale** (`no_such_ask`) within seconds | re-query every round instead of reusing one list; do not conclude "no GPUs available" |
| V53 | a log-based progress counter went DOWN and looked like a stall | every line was written to `/root/ocr.log` twice (the server appends to the file **and** the shell redirected stdout into it) | write the file in exactly one place |
| V54 | a runner's log stayed 0 bytes for an hour and looked like a crash | the child's output was captured (`capture_output=True`) and only printed at the very end | stream the child's output into the job log |
| V55 | the wrong archive was picked for a page-count lookup | two different volumes had 237 pages | identify inputs by page count **and** skip any that already have an output archive |
| V56 | a failed pilot destroyed the only debugging environment | a `finally: destroy` guard was added to guarantee "nothing left billing" | **keep the machine when a run fails** (that is exactly when it is needed); destroy only when the work is done or the user says so |
| V57 | a native crash left no explanation at all | the process died with no traceback and the container is not reachable except by SSH | `faulthandler` into a log file, per-stage markers, VRAM in `/health`, a `GET /logs` HTTP endpoint, and a restart loop in the start script |
| V58 | `ollama pull hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q6_K` cannot work | **no file in that repo contains the string `Q6_K`** - the *unsuffixed* GGUF is Q6_K (the model card says so), and the quant tag only resolves when a filename matches it | download the unsuffixed GGUF and import it with a Modelfile (`ollama create`); do not guess quant tags - list the repo's files first |
| V59 | the chosen quantisation was never justified | "Q4_K_M is the default" is a habit, not a measurement | check the community evidence: llama.cpp's perplexity table (Q4_K_M **+0.0535** vs Q5_K_S +0.0353 vs Q6_K **+0.0044**), the VRAM headroom rule ("model under 60% of VRAM → use Q6_K"), and **what the model's own authors deploy** - SakuraLLM's tutorial downloads the Q6_K file |
| V60 | a new model version shipped with the old prompt | each Sakura release is tuned with a specific system prompt; `Sakura-14B-Qwen3-v1.5` recommends a *different* one from the GalTransl line, and using the old wording degrades its output | **read the model card of the version you ship** and copy its recommended system prompt (`mth/providers.py`); changing the model without changing the prompt is a silent quality regression |
| V61 | a batch was started without checking that the model can actually translate | "the port answers" and "the model loads" do not prove the chat template or the sampling are right; a reasoning model can emit ` thinking` noise into every answer | run the pre-flight gate (translate 3 lines, require exactly 3 non-empty Chinese lines with no reasoning/template noise) before any batch |
| V62 | whole pages were delivered with the original Japanese still on them | a **single** detection pass misses whole classes of text (vertical lines, white-on-dark, low contrast). An objective audit (compare the text-stroke density of the input page with the delivered page) found **74 of 1034 pages unchanged and still text-like**, ~32 of them real text pages; one book had 20 consecutive pages missed | run the plain pass first and **escalate** to rotated (vertical text) and inverted (white-on-dark) passes when it finds suspiciously little text; map the rotated boxes back to the original coordinates and dedup by IoU. Verify with the audit script, before and after |
| V63 | a quality claim could not be checked | "the detection looks better now" is not a measurement | audit by **comparing input with delivered output** (stroke density, block counts per page); it needs no viewing, no GPU and no money, and it produced the 74-page finding |

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

**R7 Exactly one machine, ever**
22. **One instance at a time is a hard invariant.** `up` refuses while any
    instance of ours is running or loading - it does not "clean up and continue".
23. **Never write ad-hoc rental code.** Rental only happens through
    `hanhua.py vast up`. A throwaway script that can rent is how a second machine
    got started while the first was still up (two machines billing at once).
24. Record the one instance you own (`vast_current.txt`) and run
    `deploy/vast_watchdog.py` (or an equivalent) while a run is in flight: it
    destroys any instance of ours that is not the recorded one. It must never
    touch instances that lack our label.
25. If an instance you created **disappears**, STOP and report - never
    interpret it as "slow host, try the next offer" and keep renting.

**R8 Failures keep the machine**
26. A failed run **keeps** the machine. That is exactly the moment you need it
    (logs, SSH, a re-run without re-pulling 15 GB). A `finally: destroy` guard
    once threw away the only debugging environment - do not repeat that.
27. Destroy when the work is verified complete, when the user says so, or when
    the machine is provably unusable (see R7.5 / the pull watchdog).

**R9 Pick hosts that can actually pull**
28. Docker Hub throughput is **per host**. Measure the pull: if the image is not
    down within ~15 minutes, destroy and switch (a warm-cached host can boot the
    same 15 GB image in ~3 minutes).
29. Prefer hosts that already hold the image; blacklist hosts that pulled slowly.

**R10 Make transports survive and failures visible**
30. **Never send a long silent request.** A non-streaming completion leaves the
    socket idle for the whole generation and the host's port forward drops it.
    Stream it, and retry transient transport errors (both the LLM and the OCR
    client).
31. Build in diagnosability before renting: `faulthandler` to a log file,
    per-stage log markers, VRAM in `/health`, a `GET /logs` endpoint reachable
    over HTTP, and a restart loop for the service. A silent native crash is
    undebuggable and wastes a whole rental.
32. Identify work items unambiguously (two different volumes had 237 pages) -
    page count alone is not an identity. Skip inputs that already have outputs.

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
