# 05 — Machine reports (anonymised)

History of every machine rented during development. Host addresses, instance ids,
account details, and the input material are **removed on purpose**. Machines are
labelled `M1…M27`; offers that were already taken (no instance created) are
grouped.

## Summary

| Outcome | Count |
|---|---|
| Working (reached readiness and produced output) | 4 |
| Bad network (download unusably slow / stalled) | 10 |
| SSH unreachable (turned out to be a client-side key problem, see below) | 8 |
| Never reached "running" (image load timed out) | 3 |
| Offer already taken (no instance created) | 5 |
| Expensive fallback stuck loading (manually destroyed) | 1 |
| Concurrent duplicates (two controller processes) | 2 |

> The **SSH-unreachable** group was **not** a machine problem: the client's
> private key had permissions OpenSSH rejects on Windows, so every `ssh`/`scp`
> failed silently and the network check returned empty output, which the
> controller misread as "bad network". This single issue caused most of the
> machine churn. See `06_bugs_and_fixes.md` K1.

## Detail

| Machine | GPU class | Unit price | Offer-declared bandwidth | Outcome | Action |
|---|---|---|---|---|---|
| M1 | RTX 3090 | mid | high | network check **BAD** (~1 MB/s, unstable) | destroyed |
| M2 | RTX 3090 | mid | high | network check **OK**, fast downloads | **SUCCESS** (full run) → destroyed |
| M3 | RTX 3090 | low | very high | check BAD (empty output; really the key bug) | destroyed |
| M4 | RTX 3090 | low | high | never reached running | destroyed |
| M5 | RTX 3090 | low | high | never reached running | destroyed |
| M6 | RTX 3090 | low | low | check BAD | destroyed |
| M7 | RTX 3090 | low | — | replaced while loading | destroyed |
| M8 | RTX 4090 | **expensive** | — | stuck loading; cost ran up | **manually destroyed** |
| M9–M16 | V100 / 3090 / RTX 8000 | mid–high | high–very high | check BAD (empty output; key bug) | destroyed |
| M17–M22 | 3090 / V100 / RTX 8000 | mid–high | high | ssh not ready (key bug) | destroyed |
| M23 | V100 | mid | high | endpoints up; model pull failed, OCR 500 | salvaged manually, then the run aborted on a consistency check → destroyed |
| M24 | V100 | mid | very high | endpoints up, model present, erase worked | **SUCCESS** → destroyed |
| M25 | V100 | low | mid | model download **stalled at 19%** | destroyed |
| M26, M27 | RTX 3090 ×2 | mid | high | created concurrently by two controller processes | destroyed |
| M28 | V100 | mid | very high | prebuilt image; fast image pull (~38 MB/s), fast model pull (~264 MB/s), HF probe 32 MB/s | **SUCCESS** (baked image, exact dimensions, 0 violations) → destroyed |

Offers that were already taken (create returned "not available"): 5 separate
attempts across runs; no instance was created and no meaningful cost accrued.

## Patterns

1. **Advertised bandwidth and reliability do not predict success.** Several
   machines advertising multi-gigabit bandwidth and maximum reliability failed,
   and one low-priced machine was the best performer. Only on-machine measurement
   is trustworthy.
2. **The same physical host can appear as several offers.** One host's offers
   failed repeatedly; excluding failed hosts after the first failure removed the
   repeat waste.
3. **The dominant failure was client-side, not the machines.** Once the private
   key permissions were fixed and SSH was removed from the happy path, machines
   started succeeding immediately.
4. **A bad download path must be detected at boot.** The last failure was a
   machine whose model download stalled at 19%; a boot-time speed gate now makes
   the controller give up in a couple of minutes instead of thirty.

## Cost

Across both API keys, the total spent was under one US dollar, and the majority
of it was machine switching, not translation. The successful translation run cost
well under ten cents. See `11_lessons_and_guardrails.md`.

## Judging "usable"

With SSH removed, the only trustworthy readiness test is:

1. `GET /v1/models` returns a **non-empty** model list (LLM).
2. `GET /health` returns `{"ok": true}` (OCR).
3. A real `POST /prepare` returns both blocks and a cleaned image.

All three pass → usable. Anything else within a couple of minutes → destroy and
switch.
