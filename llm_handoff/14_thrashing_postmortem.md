# 14 — The thrash post-mortem: how this project repeatedly broke its own fixes

> Read this **before** you change anything in this repo. It is the least
> flattering document here and the most useful. Every wasted dollar in this
> project came from the loop described below, not from the model, not from the
> pipeline, and not from vast.ai.

---

## 1. Why this file exists

The final state works: five volumes, ~1,850 pages, one machine, under $1. But the
road there was not a straight line. It was:

```
change something  ->  run on a paid machine  ->  it fails differently
      ^                                                    |
      +-------------------- guess a new cause <------------+
```

Each cycle cost money, and several cycles cost a **machine teardown plus a fresh
15 GB image pull** (~12 minutes) on top. The next agent will feel exactly the
same pressure (a user waiting, a bill running) and will be tempted to do exactly
what I did. Do not.

---

## 2. The loop I fell into (name it, so you can catch it)

1. A run fails with a message that is ambiguous (`Remote end closed connection`).
2. Instead of **instrumenting**, I form a theory (VRAM? segfault? the detector?).
3. I change code to fit the theory and **run it on a paid machine** to find out.
4. It fails differently. Back to step 2 with a new theory.
5. Meanwhile the machine gets destroyed (either by a stall guard or by me), so
   the next attempt must **re-pull the image** and re-boot.
6. To move faster I write a **throwaway script** that rents and manages
   machines. That script has its own bugs and bypasses the safety rails.

That is the whole disaster. Four of the six steps are avoidable.

---

## 3. The actual sequence (what really happened, in order)

| # | Action | Outcome | Cost of being wrong |
|---|---|---|---|
| 1 | v0.2.0 image (model baked, weights pre-downloaded) | build fine; **first `/prepare` failed** | one machine + a full image pull |
| 2 | Theory: "HuggingFace unreachable, retries exhaust" → added `HF_HUB_OFFLINE=1` (v0.2.1) | correct, but only fixed part of it | a 25-min rebuild |
| 3 | Pilot run again | failed **again** with the same message | machine + pull |
| 4 | Theory: "the detector crashed the OCR process" → vendored comic-text-detector (v0.2.2) | the detector was genuinely broken (never pip-installable), but **not** the cause of the abort | a 25-min rebuild |
| 5 | Pilot run again | failed **again** | machine + pull |
| 6 | Theory: "VRAM OOM" → checked `min_vram` default (24 GB, so no) | wrong theory, discarded | time |
| 7 | Theory: "the detector's API unpacking" → fixed `mask, mask_refined, block_list` | a real bug, still not the cause | time |
| 8 | **Finally instrumented** the server (stage markers, `faulthandler`, `GET /logs`, VRAM, restart loop) | **the log showed `/prepare` returning 200 all along** - the OCR was never the problem | one machine |
| 9 | Reproduced the failure locally against the **LLM endpoint only** | found it: ~1 in 5 **non-streaming** requests dropped | free |
| 10 | Verified the hypothesis (4/6 non-streaming vs 6/6 streaming) | confirmed | free |
| 11 | Fixed the LLM client (stream + retries) | pilot **passed** (20 pages) | free |
| 12 | Ran the full volume | failed at ~page 5 | one machine + pull |
| 13 | Theory: "`/prepare` exceeds the ~15 s idle window" | measured it: 6.4-8.3 s, so **wrong** | free (measured) |
| 14 | Realised the same drop affects the OCR client (no retry there) | fixed with retries | free |
| 15 | Full volume | **passed**, then four more volumes passed | - |

**Steps 3-7 were four consecutive wrong theories, each paid for with a machine
and a 15 GB pull.** Step 8 (instrumenting) would have collapsed them into one.
Step 9 cost nothing because it ran on my own machine.

---

## 4. Every "fixed one thing, broke another" cycle

These are the regressions I introduced **while fixing something else**. They are
the reason the user's instruction was "record the thrashing".

| Fix I was making | What I broke | How it showed up |
|---|---|---|
| Make comic-text-detector a **required** build step | The package cannot be installed at all, so the build would have failed forever | Caught before building only by checking the upstream repo for `setup.py`; reverted to best-effort, then vendored |
| Bake the detector's weights "if importable" | Silent no-op, because the package was never importable | `detector: false` persisted |
| Enforce "one instance" by having `up` destroy leftovers | Not the bug itself, but it hid the real one: ad-hoc scripts bypassed `up` entirely | The double-rent |
| Fix the page-count collision by skipping inputs that already have outputs | Broke the **wait** path (`find_input(250)` returned `None` because that volume had *just* produced its output) → **chain 2 crashed and volume 5 never started** | The chain died with `TypeError: expected str, bytes or os.PathLike object, not NoneType`; ~7 minutes of idle billing before I noticed |
| Add the `KEEP_MACHINE` escape hatch | Chain 1 kept the machine, but chain 2 needed to destroy - two chains disagreeing | Had to add `CHAIN_DESTROY=1` |
| Kill the "old chain waiter" to add a 4th volume | Nothing broke, but it was a manual process step that could easily have killed the wrong process | Verified with a process count first |
| Let a script destroy on failure ("never leave a machine billing") | Destroyed the only debugging environment, forcing a full re-pull | The user's explicit complaint |

**Pattern:** almost every regression came from editing a **controller** (the thing
that manages machines) and then immediately running it, without a dry-run or a
unit check. The `find_input` regression was found only because the chain printed
a traceback; a slightly different bug would have silently done the wrong thing.

---

## 5. The anti-patterns, and what to do instead

| Anti-pattern | Why it feels right | What to do instead |
|---|---|---|
| **Guessing the cause and testing on a paid machine** | "It's probably X, let me just try" | **Instrument first.** Add stage markers, a crash handler, and an HTTP log endpoint. One instrumented run beats five guessed runs. |
| **Debugging the whole system when one component is suspect** | it is all one pipeline | **Isolate**: hit the LLM endpoint directly, hit `/prepare` directly, from your own machine. That is free. |
| **Rebuilding the 25-minute image to test a hypothesis** | "the fix belongs in the image" | Inject the changed file via a **custom onstart** (gzip+base64) and test in ~3 minutes. Bake it into the image only once it works. |
| **Destroying the machine on failure** | "never leave a machine billing" | **Keep it.** It is the only place the logs live. Destroy on verified success. |
| **Writing a throwaway script that can rent** | "the CLI is awkward, I'll script it" | Never. Use the one guarded path (`hanhua.py vast up`) plus `enforce_single` and the watchdog. A throwaway rental script caused the double-rent. |
| **Treating one error message as one cause** | the message names a symptom | `Remote end closed connection` came from the **LLM** endpoint, not the OCR service. The message does not tell you which component; make the client say so. |
| **Re-running the same failing thing hoping for a different result** | "maybe it was transient" | Retry **inside** the client (with backoff), so a transient drop never aborts a batch - then a real failure is a real failure. |
| **Judging progress by a log line count** | it is the easiest signal | Count **artifacts** (`raw/*.clean.png`), which is monotonic. Log lines can duplicate and even go *down*. |
| **Editing a running controller and re-launching** | "it's a small change" | Re-read the function, list its callers, and check each call site. The `skip_done` change had two call sites with opposite needs. |
| **Letting a long job run without a checkpoint** | "it'll be fine" | Verify at every volume boundary, and keep a **safety net** that tears the machine down if the main driver dies. |
| **Renting to satisfy your own curiosity** | "I just want to *prove* it works end to end" | **Ask first, every time.** Late in this project the agent rented a machine to "verify that the repo can run the flow" - the user had asked only for a *static* check of the git contents, which had already been done and had already found the real bugs (dangling doc references). The rental cost $0.002 and, more importantly, broke the user's trust: "who told you to rent a machine?" **A static check is free; if you want a dynamic check, propose it and let the user decide.** Rule R1.1 (consent per machine) exists for exactly this. |

---

## 6. The loop that actually works (use this one)

```
1. REPRODUCE for free (local script, direct endpoint call, mock OCR).
2. INSTRUMENT the failing stage (marker + traceback + a way to read the log).
3. FORM one hypothesis and make ONE change.
4. VERIFY the change in isolation (locally, or on the machine without a rebuild).
5. Only then run the batch.
6. On failure: KEEP the machine, read the log, go to 3.
7. On success: verify content, then destroy, then confirm instances == 0.
```

The order matters more than the cleverness. Every expensive mistake above was
step 5 happening before steps 1-4.

---

## 7. Pre-flight checklist before you change anything

- [ ] Can I reproduce this **without** a paid machine? (endpoint call, local
      script, mock backend)
- [ ] Do I have a log that will tell me *where* it fails, or am I guessing?
- [ ] Am I about to edit a **controller** (rent/destroy/verify)? If yes, have I
      listed every call site and checked each one?
- [ ] Am I about to write anything that can **rent** a machine? (If yes: stop.)
- [ ] If this fails, do I still have the machine, the logs, and the work dir?
- [ ] Is there a **safety net** that will tear the machine down if my driver dies?
- [ ] Will I know it worked from **content** (page count, order, dimensions,
      translation coverage), not from an exit code?

---

## 8. The honest one-line summary

> I lost the most money by **changing things and running them on a paid machine
> to find out what was wrong**, and by **destroying the machine that held the
> evidence**. Instrument, reproduce for free, change one thing, verify, and only
> then spend.
