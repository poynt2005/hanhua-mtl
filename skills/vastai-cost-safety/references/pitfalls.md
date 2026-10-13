# Pitfall table (symptom → cause → fix)

Check this first when something fails. Source: real GPU-automation tasks where
money was lost to repeated mistakes.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| V01 | non-ASCII `print` raises `UnicodeEncodeError` on Windows | console codepage | reconfigure stdout to UTF-8; `chcp 65001` |
| V02 | line counts double after writing files | Windows text mode writes `\r\n` | write with `newline="\n"` |
| V03 | `~/.ssh` not writable | external dir permissions | keep keys in the workspace; delete after |
| V04 | `ssh-keygen -N ""` fails on Windows | empty passphrase + interactive prompt | generate the key in Python |
| V05 | instance-create output fails to parse | `"Started. {...}"`, not pure JSON | regex the braces; retry the status call |
| V06 | `Permission denied (publickey)` | **private key ACL too open → ssh ignores the key**; or key not propagated yet | `icacls <key> /inheritance:r /grant:r "%USERNAME%:F"`; wait for ssh |
| V07 | pulled a lower quantization than wanted | the repo has no file literally named after it (no suffix = one specific quant) | trust the **file size** |
| V08 | remote download 403 / key error | datacenter IPs are rate-limited | **do not call APIs from the remote** |
| V09 | setup "succeeded" but did nothing | `cmd \| tail` always exits 0 | write a log + status marker and verify |
| V10 | offer query returns nothing for a VRAM filter | that field is not queryable | drop the filter |
| V11 | translation line-count mismatch | model merges/splits newlines | retry, leave blank, verify gaps, continue |
| V12 | bill keeps growing | per-second billing during debugging | max price + auto-destroy; report money first |
| V13 | `pip install vast-ai` not found | the package is `vastai` | install the right name |
| V14 | limited shell tools, quoting hell | platform tooling | write a file and run it |
| V15 | native binary won't start | newer glibc required than the base image has | use a compatible image or a different server |
| V16 | unpacking an archive has no binary | it contains only shared objects | pair it with the real binary archive |
| V17 | a CLI entry point not found | not on PATH in non-interactive ssh | call the library directly |
| V18 | package installed but lib not found | no LD path in non-interactive ssh | set `LD_LIBRARY_PATH`, correct order |
| V19 | health endpoint 404 → server wrongly declared dead | that build has no such endpoint | probe the model list |
| V20 | every request 500, empty output | invalid chat template / parameter name | use the right one; test one sentence first |
| V21 | output file full of empty lines | a dead server appended blanks | delete old output first; count non-empty |
| V22 | remote command syntax error | the outer tool ate the quotes | write remote commands to `.sh` |
| V23 | `ssh 'nohup cmd &'` hangs locally | remote process holds the ssh fds | redirect all three fds to a log / `/dev/null` |
| V24 | GB-size file over proxy scp is unusable | relay throttling | download big files on the remote |
| V25 | destroy says `Aborted`, machine stays up | CLI prompts `[y/N]` | pipe `y` in, then verify |
| V26 | `Invalid user key` | key pasted wrong | keys are 64 chars; check length |
| V27 | GPU 0%, no model, still billing | idle | destroy after 15 minutes idle |
| V28 | overlapping installs | new command sent too early | `pgrep -af` first |
| V29 | network slow/erratic | bad host | check within 5 minutes; switch |
| V30 | one 403 treated as final | single error as conclusion | retest 3×; cite evidence |
| V31 | artifacts lost after destroy | nothing pulled first | pull and verify before destroying |
| V32 | cross-platform tool reports all-pass/all-fail | `select()` on a pipe unsupported on Windows; init events contain the skill name | validate on Windows; judge only real tool calls |
| V33 | structured-extraction task returns schema keys | a translation model is not a general instruction model | don't use it for JSON |
| V34 | a package is not found on the index | it lives on a code host, not the package index | install from the code host, or use a fallback |
| V35 | a library reports another is missing though installed | version mismatch (a newer release needs a newer framework) | pin compatible versions |
| V36 | an upload endpoint 500s on first call | missing multipart helper | install it |
| V37 | a CLI flag silently ignored | defined both globally and per-subcommand | define it once |
| V38 | output image size/quality drops | default JPEG re-encode quality + chroma subsampling | save with the source quality and no subsampling |
| V39 | standing up the LLM server is painful | binary/glibc and chat-template issues | use a server that handles templates (e.g. ollama) |
| V40 | a stalled model pull looks "ready" | the server answers 200 with an empty model list | readiness requires a **non-empty** model list |
| V41 | a bad-network machine wastes 30 minutes pulling | no speed gate | probe the download speed at boot and fail fast |
| V42 | a prebuilt image boots but its services never start | the instance runs its own entrypoint (sshd), so the image `CMD` is not executed | pass an onstart command that runs the start script |

## Four root classes

1. **Failures are hidden** (V09/V19/V21): judge by content and markers.
2. **Environment assumed** (V15–V20): verify external deps locally; test one sentence.
3. **Wrong order** (V20/V21): gates before batches.
4. **Tooling friction** (V14/V22/V23): write `.sh` files, redirect fds.

Plus: **fixing a bad machine** (V24/V29) and **stopping without revoking** —
both are judgement errors; switch machines, and stop = destroy + revoke.
