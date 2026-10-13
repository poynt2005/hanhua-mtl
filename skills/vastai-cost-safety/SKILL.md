---
name: vastai-cost-safety
description: >-
  Keep rented-GPU work from burning money on vast.ai (and any hourly cloud GPU).
  Load it BEFORE you rent, create, SSH into, debug, or destroy a remote GPU
  instance — and whenever a task involves cloud GPU, renting a machine, hourly
  GPU billing, downloading model weights, running a setup script on a remote box,
  SSH/SCP to a rented instance, or "why is my instance still charging". It covers
  machine selection, the 5-minute network check, never-upload-weights, detecting
  fake-success setup, the three pre-run gates, billing caps, and guaranteed
  teardown. If GPU compute is billed by the hour, read this first — even if the
  user never says "vast.ai".
compatibility: Requires the `vastai` CLI (pip install vastai) and plain bash for the bundled scripts. Network checks run on the remote host.
---

# vast.ai cost safety

## Why this skill exists

GPU time is billed by the second, and **failures disguise themselves as
success**. This skill comes from real tasks where machines sat idle and money
burned with no output — almost all of it spent on unvalidated setup scripts run
for the first time on a paid box.

**One line**: prove it is alive, translate one sentence, only then run a batch;
judge by content not line counts; write remote commands to files; always verify
teardown.

## Three highest rules

1. **Report the budget and get explicit consent before renting.** No consent = do not rent.
2. **Judge success by content and marker files**, not line counts or exit codes.
3. **On completion / stop / handoff / leaving: destroy first and verify.**

---

## Money-burning behaviours (stop if you catch yourself doing these)

| Behaviour | Why it burns money | Do instead |
|---|---|---|
| Renting without explicit consent | billing starts immediately | report "N × $X/hr × Y hr ≈ $Z", then rent |
| Iterating on setup on a paid machine | every minute is billed | write and syntax-check scripts locally first |
| **`scp`-uploading model weights or big wheels** | relay/proxy throughput is terrible | download on the remote (`ollama pull`, `pip install`) |
| Not checking the network before setup | a bad host wastes the whole run | run the network check within 5 minutes; switch if bad |
| Debugging a bad host (new proxy, CPU fallback, retry) | all of it is still billed | destroy and switch machines |
| Launching a batch without a single-sentence test | a bad parameter fails everything | three gates: alive → one sentence → no stale process |
| Trusting `cmd \| tail` | the exit code is always 0 | write a log and a status marker, then verify it |
| Probing the wrong health endpoint | a live server looks dead | probe the model list, not `/health` |
| Counting total lines as progress | a dead server keeps appending blanks | count non-empty results |
| Overlapping runs of the same job | interference, duplicate downloads | `pgrep -af` before a new command |
| GPU idle 15 minutes with no download | pure waste | destroy |
| Fixing the same error a third way | keeps burning | stop and report |
| Destroying before pulling artifacts | the work is gone | pull and verify first |
| `destroy` without confirming the prompt | the CLI waits at `[y/N]` and the machine stays up | pipe `y` in, then verify `show instances` |
| Concluding from a single 403 | a wrong conclusion causes churn | retest 3×, cite evidence |
| Exceeding the task budget | runaway cost | set a cap; report at 80%, stop at 100% |
| Leaving a machine unattended | keeps billing | destroy when you stop watching |
| Writing API keys to files | leak / theft | environment variables only; revoke after |

---

## Before renting

1. User consent + budget reported.
2. Machine selection: reliability floor, `--direct`, max price.
3. Scripts written and syntax-checked locally.

## Within 5 minutes of boot

```bash
scp -i <key> -P <port> scripts/net_check.sh root@<host>:/root/net_check.sh
ssh -i <key> -p <port> root@<host> "bash /root/net_check.sh" </dev/null
```
Only the sources you actually need (package index, model host) are required;
other sites (often rate-limiting datacenter IPs) are informational. Also time a
10MB scp from your machine; **> 30s → switch**. `NET_BAD` → destroy and switch;
three bad machines → stop and report.

## While running: three gates

```
1) alive   : the model list endpoint returns a model
2) one test: a single-sentence request returns 200 with sane output
3) no stale process: pgrep -af shows nothing from a previous run
```

## Teardown

1. Pull artifacts back and verify.
2. Destroy and confirm `show instances` is empty and billing stopped.
3. Remove the key from the environment.

## Reporting format

First line is always the money:

```
instances: N | total_spend: $X | this task: $Y
- machine <id> (<gpu>, $<rate>/hr) network: OK / BAD
- decision: kept / destroyed and switched
```

---

## Quick facts

- **Big-file rule**: weights are never uploaded; download on the remote; only KB-size scripts are copied.
- **Do not call APIs from the remote**: datacenter IPs are often rate-limited; precompute names/URLs locally.
- **Verify the tool**: the package is `vastai`; API keys are 64 chars; some offer fields are not queryable.
- **Windows**: reconfigure stdout to UTF-8; write files with `newline="\n"`; **lock the SSH key ACL with `icacls`** or ssh silently ignores the key.
- **Model files**: trust the file size over the GGUF name.
- **Bake the environment into an image** and smoke-test it at build time — that moves the whole class of "discovered on a paid machine" failures to a free step.

## References

- `references/pitfalls.md` — the full pitfall table (symptom → cause → fix).
- `references/hard-rules.md` — the hard rules in full.
- `scripts/net_check.sh` — network check (`NET_OK` / `NET_BAD`).
- `scripts/safe_destroy.sh` — destroy and verify billing stopped.
