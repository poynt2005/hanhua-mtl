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
