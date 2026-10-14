# 06 — Bugs and fixes

Every defect encountered during development, with the fix. IDs are referenced
from other documents.

## A. Client / Windows

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| K1 | every `ssh`/`scp` failed silently; network checks returned empty output | the generated private key's ACL was too open, so **OpenSSH on Windows ignored the key** (`UNPROTECTED PRIVATE KEY FILE`); `os.chmod` does not change Windows ACLs | run `icacls <key> /inheritance:r /grant:r "%USERNAME%:F"` after generating the key |
| K2 | a subprocess reader thread raised `UnicodeDecodeError` | capturing a tool's output without specifying an encoding (system codepage) | pass `encoding="utf-8", errors="replace"` to `subprocess.run` |
| K3 | a CLI flag was silently ignored | the option was defined both globally and per-subcommand, and the subcommand default overwrote it | define the option once |
| K4 | non-ASCII text mangled when piped into a script | console/stdin codepage | write a file instead of piping, or set UTF-8 mode |

## B. GPU stack installation (all previously hit on a paid box)

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| G1 | `pip install comic-text-detector` not found | it is **not on the package index** (it is a code-host project) | install from the code host; otherwise fall back to a general detector |
| G2 | OCR model reported "PyTorch not found" though torch was installed | a newer `transformers` release requires a newer torch than the image had | pin a compatible `transformers` |
| G3 | OCR endpoint returned 500 on the first call | `cv2` could not import because `libGL.so.1` was missing | `apt-get install libgl1 libglib2.0-0` |
| G4 | upload endpoint returned 500 | the multipart helper was missing | install it |
| G5 | the model was never pulled | the model CLI talked to the wrong host/port (its default) instead of the running server | set the server host for CLI commands |
| G6 | the LLM server did not survive | background processes started during provisioning needed all three fds redirected | `nohup ... >log 2>&1 </dev/null &` |
| G7 | output image dimensions changed | the inpainting model pads to a multiple of 8 and the result was not cropped back | crop to the original size |
| G8 | a stalled model download looked "ready" | the LLM server answers 200 with an empty model list | readiness requires a **non-empty** model list |
| G9 | on a baked image, the **first** `/prepare` failed (`Remote end closed connection`) though `/health` worked | `transformers` performs a Hub *freshness* check on every model load even when the weights are cached; on a host that cannot reach HuggingFace it retries until the request fails | bake the weights into fixed cache paths and force the runtime offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) |
| G10 | text detection silently degraded to EasyOCR boxes; `detector` was always `false` | the detector **cannot be pip-installed at all** (no `setup.py`/`pyproject.toml`, not on PyPI, top-level imports) and its weights are not in the repo; `get_detector()` also hard-coded `device="cuda"`, and `detect()` unpacked the wrong return value (`mask, groups, _`). Every failure was swallowed by a broad `except` | vendor the repo at a pinned commit + fetch the weights; install `pyclipper`/`shapely`/`torchsummary`; make the training-only `wandb` import optional; pick the device from `torch.cuda.is_available()`; use the real return value `(mask, mask_refined, block_list)`; assert the detector in the build smoke test |
| G11 | every pilot aborted mid-run with `Remote end closed connection` / `WinError 10054`, looking like the OCR service had crashed | the OCR service was fine — **the LLM request was the problem**. A non-streaming completion is silent for the whole generation, and the GPU host's port forward drops an idle TCP connection after ~15 s (measured: 4/6 non-streaming requests succeeded, **6/6 streaming**). The stage markers proved `/prepare` returned 200 throughout | send LLM requests with `stream: true` and reassemble the SSE deltas, plus bounded retries (`config.stream`, `config.retries`) |
| G12 | a crashed service gave no explanation at all | the process died with no traceback, stderr was redirected away, and the container is not reachable except by SSH | `faulthandler` into a log file, per-stage log markers, VRAM in `/health`, a `GET /logs` endpoint (readable over HTTP after a crash), and a restart loop in the start script |
| G13 | a log-based progress counter went DOWN, looking like a stall | every line reached `/root/ocr.log` twice (the server appends to the file **and** the shell redirected the service's stdout into it) | write the log file in exactly one place |
| G14 | a runner's log stayed 0 bytes for an hour, looking like a crash | the child's output was captured (`capture_output=True`) and printed only after each volume | stream the child's output into the job log |
| G15 | `ollama pull hf.co/SakuraLLM/Sakura-GalTransl-14B-v3.8:Q6_K` cannot resolve, and the quality of the shipped quant was never justified | no file in that repo contains the string `Q6_K` (the *unsuffixed* GGUF is Q6_K), and the quant was inherited from the "Q4_K_M is the default" habit rather than measured | download the unsuffixed GGUF and `ollama create` from a Modelfile; choose the quant from evidence (llama.cpp perplexity table, the VRAM-headroom rule, and what the model's own authors deploy - SakuraLLM's tutorial uses Q6_K) |
| G16 | whole pages were delivered with the original Japanese still on them, while every check still passed | a **single** detection pass misses whole classes of text (vertical lines, white-on-dark, low contrast). Page count/order/dimensions and "100% of *detected* blocks translated" all looked perfect. An objective audit (input page vs delivered page, by text-stroke density) found **74 of 1034 pages unchanged and still text-like**, ~32 of them real text pages; one volume had **20 consecutive pages** missed | plain pass first, then **escalate** to rotated (vertical) and inverted (white-on-dark) passes when it finds fewer than 5 boxes; map rotated boxes back to the original coordinates (unit-tested) and dedup by IoU; `DET_SIZE` 1024 -> 1280 |
| G17 | G16's "74 pages" figure was an over-call, and a fix was built and measured against it before anyone looked at the pages | the stroke-density metric **cannot tell dialogue from onomatopoeia / SFX**, and this pipeline deliberately keeps SFX (`keep_sfx: true`). The adaptive detection was built, shipped and measured (30 -> 29 low-block pages, 23 -> 23 still text-like - i.e. no real movement) before the user looked at the contact sheets and said the pages were almost all SFX | produce contact sheets (`tools/audit_sheets.py`) and have a human check them **before** changing code; treat an automated metric as a triage list, not a defect count. The adaptive detection was kept (it is harmless and found +49 blocks) but it is **not** claimed as a fix |

## C. Controller logic

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| V1 | a pre-run cleanup would have destroyed **all** instances, including the user's others | no label filter | destroy only instances labelled by this tool |
| V2 | a machine leaked and kept billing | the rent loop had no exception handling | wrap the loop body; always destroy on an unexpected error |
| W1 | a bad-network machine wasted up to 30 minutes | no download-speed gate | probe speed at boot, expose it via `/health`, fail fast below a threshold |
| W2 | SSH readiness was assumed | the key needs time to propagate; direct ports may be closed | (superseded) the happy path no longer uses SSH |
| S1 | the controller picked expensive machines | it sorted by reliability first | sort cheapest-within-a-reliability-band; enforce a max price |
| S2 | the network check failed almost every machine | it treated an unrelated site as required, and that site rate-limits datacenter IPs | only require the sources actually used |
| D1 | two machines billed at once | two controller processes were running (an earlier one had not exited) | destroy leftovers before starting; ensure only one controller runs |
| C1 | a baked image booted but its services never started | a cloud SSH instance runs its **own entrypoint** (sshd), so the image's `CMD` is **not executed** | pass an `--onstart-cmd` that runs the baked start script (do not rely on the image `CMD`) |
| C2 | good machines rejected because the model download was slow | the boot flow pulled the model from HuggingFace, which throttles datacenter IPs (measured ~4 MB/s) | **bake the model into the image**; readiness no longer depends on HF |
| C3 | healthy machines were destroyed during boot | the image-pull watcher treated "no status change for 4 minutes" as a stall, but a 9 GB layer (and the unpack phase after it) changes no status for minutes | only give up on a pull that is repeatedly **retrying**, and allow a generous total timeout |
| C4 | every `create instance` was refused, with no reason shown | vast rejects a create when the whole `args` string exceeds **16384 chars**, and the injected server payload was ~19.7 KB | gzip+base64 the injected files (~7.9 KB) and refuse to send an oversized command |
| C5 | a whole rental round looked mysterious | `create_instance()` discarded vast's stderr, so the real message (`no_such_ask`, args-too-long) was invisible | surface the refusal reason when the JSON reply cannot be parsed |
| C6 | ~60 minutes of billing on one host that never finished booting | Docker Hub throughput is **per-host**: one host pulled at 3.6 MB/s (16% in 12 min), another had the layers warm and reached `running` in 187 s with the same image | add a pull watchdog (give up after N seconds), prefer hosts that already hold the image, and blacklist the slow host |
| C7 | a hard-won machine was destroyed automatically after a failed pilot | the runner had `finally: destroy` to guarantee "nothing left billing" | keep the machine when a run fails (that is exactly when it is needed for debugging); destroy only when the work is done or the user says so |
| C8 | **two machines billing at once** | a leftover ad-hoc script read "my instance vanished" as "slow host, try the next offer" and kept renting; it also bypassed the blacklist and the leftover cleanup that `vast up` does | single-instance invariant: `up` refuses while one of ours is running/loading; `enforce_single()` + a recorded `vast_current.txt`; a watchdog destroys extras; delete ad-hoc rental scripts |
| C9 | every `create instance` was refused with no visible reason | vast rejects a create when the whole `args` string exceeds **16384 chars** and the injected start payload was ~19.7 KB | gzip+base64 the payload (~7.9 KB); refuse to send an oversized command |
| C10 | a rental round looked like "no machines available" | `create_instance()` discarded vast's stderr (`no_such_ask`, "Invalid args") | surface the refusal reason when the JSON reply cannot be parsed |
| C11 | offers were refused seconds after being listed | vast's bundle list goes **stale** (`no_such_ask`) | re-query the offer list every round; do not conclude "no GPUs available" |
| C12 | ~60 minutes of billing on a host that never finished booting | Docker Hub throughput is per host (one pulled at 3.6 MB/s = 16% in 12 min; another with warm layers reached `running` in 187 s) | pull watchdog + prefer hosts that already hold the image + blacklist the slow host |
| C13 | the wrong archive was selected for a page-count lookup | two different volumes both had 237 pages | identify an input by page count **and** skip anything that already has an output archive |
| C14 | a machine was rented for an image that did not exist yet | a run was launched while the CI build was still going; vast booted an instance that reported `manifest for <repo>:<tag> not found`, and the controller would then wait minutes for a boot that could never happen | **check the registry tag exists before renting** (`hub.docker.com/v2/repositories/<repo>/tags/<tag>`); the pilot script now aborts instead of renting |
| C15 | a host pulling at 1.1 MB/s kept passing the pull watchdog | the watchdog judged "did a layer complete recently" and "is it retrying"; **small layers completed every few minutes** while the 12 GB layer crawled, so neither rule fired, and the host was kept for 19 minutes | judge the **measured MB/s and the projected ETA** (`tools/pull_watch.py`), not just activity |

## D. Quality

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| Q1 | erased text was incomplete; output looked overlaid | erase used whole rectangles / a colour fill, and the boxes were tight | use a neural inpainting model with a **text-based mask** (Otsu inside each box, dilated) |
| Q2 | the text area was redrawn including bubble borders | the mask was the whole box | same as Q1 |
| Q3 | output image size and quality dropped a lot | the default JPEG re-encode used low quality plus chroma subsampling | save with the source quality and `subsampling=0` |
| Q4 | the translation read like a literal machine translation | a simplified prompt, no context, and very large batches | use the model's tuned prompt, pass recent context, and use small batches with the recommended sampling |
| Q5 | structured extraction returned schema keys | a translation model is not a general instruction model | do not use it for JSON tasks |
| Q6 | a failed run deleted the cleaned pages | the temp work dir was removed in `finally` even on failure | keep the work dir on failure; `--keep-work` keeps it on success too |
| Q7 | term consistency check aborted a whole run and lost work | the run aborted on any inconsistency | warn and continue by default; `--strict` to abort |

## E. Ordering / process (the expensive lesson)

| ID | Symptom | Root cause | Fix |
|---|---|---|---|
| P1 | repeated wasted runs | setup scripts were executed for the first time on a paid machine | bake the environment into an image and smoke-test it at build time |
| P2 | repeated machine churn | the controller destroyed a working machine before the output was confirmed | keep the machine until the output is confirmed; reuse it for fixes |
| P3 | unclear failures | no logs were reachable | surface errors in HTTP responses; keep work dirs |
