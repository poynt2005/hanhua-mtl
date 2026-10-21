# 18 — The whole journey, for the next agent

A chronological account of the second batch, written so that an agent arriving
with no context can understand **what was tried, what broke, what it cost, and
what is left**. Everything here is count-and-behaviour level: no input titles, no
page contents, no host addresses, no keys. For the per-archive numbers see
`17_run_report_batch2.md`; for the plan as designed before the run see
`16_batch_operations.md`.

## 0. Where things stand right now

| | |
|---|---|
| delivered + verified | **31 archives / ~4,080 pages** |
| still to translate | **7 archives / ~2,100 pages** |
| machines running | **0** (last one destroyed on request) |
| spend for the batch | **~$9**, of which ~$3.2 was operator error |
| operator's local API route to `console.vast.ai` | **broken** at the time of writing (see §7) |

Resume steps are in §9. Read `deploy/VASTAI_PLAYBOOK.md` before renting anything.

## 1. What the material turned out to be

The batch was 43 archives / 6,557 pages. Four were already translated into
Chinese and were excluded after opening pages (one Simplified MTL, three
Traditional) - 281 pages, no work needed.

The rest is not manga in the usual sense. A large share are CG collections:

* the same drawing ships **twice**: once with dialogue printed straight onto the
  artwork, once clean;
* there are no speech bubbles, so "erase the text" means inpainting over art;
* page counts are large (200-380) and text density is high (8-9 blocks/page).

Three pairing layouts were observed: explicit `…_t<n>` / `…_n<n>` markers, an
appended clean half, and interleaved pairs. This drives several decisions below.

## 2. Decisions that were taken before spending anything

1. **Do not rebuild the image.** The tiled model + OCR + LaMa stack was already
   baked and had delivered a previous batch; every remaining problem was client
   side or operational.
2. **Do not add concurrency.** It is the only lever that reduces GPU-hours, but it
   multiplies failure modes already paid for (truncated context, model reloads,
   pages swapped between translations). Machines are cheaper than bugs.
3. **Do not re-quantise.** Quality must stay comparable to the previous batch.
4. **Textless pages are a first-class case**: they must not be translated and
   must not be re-encoded.
5. **One thing at a time.** Every change was validated offline (mock OCR, local
   render) before it touched a rented machine.

## 3. The five pipeline changes this batch forced

| change | why | how it was verified offline |
|---|---|---|
| `list_images` collects **every** page extension | a single glob silently dropped a `.webp` beside `.jpg` pages, then repack aborted with `translated page missing` | 21/21 and 134/134 page round trips, identical names/order/dimensions |
| textless pages are **copied from the source** | the pass-through was conditional on "no cleaned copy exists", but the OCR service always made one - baking in the detector's false positives | 12 of 24 pages byte-identical by hash |
| overlapping boxes **merge before translation** | on text-on-art the detector returns a tight box and a loose box for one column; translating both printed two sentences on top of each other | 91 blocks -> 59 on a real page set; re-render readable |
| boxes over **20 % of the page are dropped** | on clean art the detector fires on texture and OCR invents a short line; typesetting it filled the page with characters | reproduced the defect and the fix on real pages |
| the model is **unloaded before OCR** | a 14B Q6_K holds 13.8 GB of a 23.6 GB card, leaving too little for LaMa | remote VRAM 22.9 GB -> 3.0 GB during detection |

Two of these were rejected after measurement and are worth remembering:

* **Pairing text pages to their clean twin by image similarity does not work.**
  Inside a single-scene CG set every clean page resembles every printed page:
  true-pair MAD p50 18.3 vs best false candidate 19.2 (24-page set), 10.7 vs 12.1
  (221-page set). Pairing is therefore by filename only (`_t<n>` / `_n<n>`), which
  measured 109/109 correct; everything else keeps the erase path.
* **Detecting textless pages statistically does not work either.** Per-page OCR
  volume does not split cleanly enough to be trusted; a wrong split leaves
  Japanese untranslated, which is worse than a small extra line. Clean ranges are
  declared per archive (`skip_pages`) or not at all.

## 4. What the rented machine taught, in order

1. **Image pull is not billed.** Only storage is charged until the instance is
   `running`. Two hosts were destroyed early as "slow" for no financial gain.
2. **Pull progress is in `status_msg`**, layer by layer. There is no byte counter,
   but layer completion plus the registry manifest gives a real rate.
3. **The console's "Mbps" is advertised bandwidth**, not live traffic: the renter
   API has no per-instance byte counters at all (verified by scanning the console
   bundle). Live numbers come from `/proc/net/dev` on the box.
4. **`vastai execute` is not always available.** On one host every command,
   including `echo hi`, returned `invalid_args`. SSH worked there. Do not build a
   monitoring plan that assumes only one of the two.
5. **A scoped API key cannot be minted with a key that lacks `user_write`** (401
   from `/api/v0/auth/apikeys/`). The dead-man switch that depended on one was
   therefore never armed; the local scheduled guard covers that role instead.
6. **The OCR service's VRAM never comes back.** It climbs from 0.4 GB to 17 GB
   over a long detection pass. With a model on top, translation stalls on a
   request that can never be served. The pipeline now stops the service after
   extraction and restarts it before the next archive.
7. **`pkill -f ocr_server.py` kills the shell that sends it**, and with it the
   restart loop, because both command lines contain the pattern. The loop now
   lives in its own script and the kill uses `pkill -x -f` on the exact child.
8. **A 34.8 MP page cannot be inpainted on 24 GB.** LaMa asks for 25.45 GiB. The
   service answers with blocks but no cleaned image, and the client used to fall
   back to blurring the text. That archive needs a bigger card or a server-side
   downscale and is out of scope for this run.

## 5. The hard incidents, in the order they happened

1. **Three rentals died before one booted.** Offer churn (`no_such_ask`), one host
   that never started, one that pulled 18.8 GB in 13 minutes and then never
   reached `running`. Pulling was free; only wall clock was lost.
2. **The first pilot failed on quality, not cost.** Pages with no text came back
   with page-sized characters printed over the art, because the detector had
   hallucinated a box covering most of the page. Fix: the 20 % area cap, plus an
   explicit `skip_pages` list for the known clean half.
3. **The same pilot used blur instead of inpainting** for one archive: LaMa ran
   out of memory and the client silently degraded. Fix: release VRAM before OCR;
   that archive remains out of scope (see §4.8).
4. **The local runner was killed twice by the environment restarting.** The second
   time the machine billed for **six hours** unattended (~$1.29). At that moment
   no supervisor existed: the guard and the monitoring subagent were both written
   the next morning. This is the single most expensive lesson of the session.
5. **A healthy run was killed as "stuck"** while the GPU was at 98 %. The probe
   used to judge it was a chat request queued behind the in-flight one (`-np 1`),
   so it measured queue time, not model speed. Rule adopted: judge by GPU
   utilisation and file writes, never by a latency probe during a batch.
6. **A "configuration improvement" broke quality.** The unload was made
   conditional on large pages; normal pages then shared the card with a 17 GB OCR
   service and inpainted badly. Two archives were re-run. Rule adopted: do not
   change a running configuration mid-batch.
7. **Two `run_list` processes were not two runners.** The venv's `python.exe` is a
   shim that re-execs the real interpreter, so every script appears as a
   parent/child pair. Acting on the process count killed a healthy batch; ~12
   minutes were redone. Rule adopted: count chains, inspect the leaf.
8. **The machine was destroyed once against an explicit instruction to keep it.**
   The command had already been sent when the instruction arrived. Nothing was
   lost except the rental, but it is the clearest example of why teardown must be
   an explicit, confirmed action.
9. **The operator's route to `console.vast.ai` died** while the machine stayed
   reachable. The first answer - "we cannot check the balance or tear down" - was
   wrong: `cloud.vast.ai` serves the same API on the same paths, so setting
   `VAST_API_BASE=https://cloud.vast.ai` restored status, teardown and the guard.
   The lesson is not that the network failed, it is that an unreachable host is
   not the same thing as an unavailable service.

## 6. Cost, decomposed

| item | ≈ |
|---|---|
| work that produced delivered pages | $5.8 |
| six hours of idle billing | $1.29 |
| re-runs caused by the four judgement errors | $0.5 |
| early phase before the VRAM fix (35-60 s/page) | part of the $5.8 but at 3-4x the marginal rate |

Marginal cost in steady state: **≈ $0.0008/page**. Session average including all
mistakes: **≈ $0.0022/page**. The difference is the price of learning on a paid
machine, and it is the reason the playbook now has V79-V82.

### 6.1 Why the waste happened: one pattern, five times

| error | root cause | countermeasure now in place |
|---|---|---|
| six hours of idle billing ($1.29) | no supervisor existed at that moment: guard and subagent were written the next morning, and the operator only wakes on a *completed* background command | scheduled `HanhuaGuard` + heartbeat refreshed every 30 s, tested end to end |
| early phase at 35-60 s/page ($1.2) | the OCR service's VRAM growth was never measured; the card was silently oversubscribed | stop OCR after extraction, restart before the next archive, unload the model before detection |
| killed a healthy run as "stuck" ($0.13) | judged on a chat latency probe that was queued behind the in-flight request (`-np 1`) | judge only on GPU utilisation, power draw and file writes |
| "unload only for big pages" broke inpainting ($0.15) | a working configuration was changed mid-batch without re-validating | changes happen between runs, never during one |
| `pkill` self-match, twice ($0.15) | the kill pattern also matched the shell and the restart loop sending it | dedicated loop script + `pkill -x -f` on the exact child |
| venv shim read as "two runners", batch killed ($0.05) | counted processes instead of inspecting the chain | count chains, check the leaf's GPU/file activity before acting |
| machine destroyed against instruction ($0.10) | teardown was treated as routine | teardown is an explicit, confirmed action |

Five of the seven share one shape: **act on a single signal without measuring an
independent one.** The durable fix is not "be more careful" - it is to make the
decision rule mechanical: a batch is working if the GPU is drawing power and files
are appearing; a machine is finished if the run log says verified; money stops
when a heartbeat stops, not when someone notices.

Cost compared with the first batch: that one measured ~$0.00064/page at
$0.153/hr. This one runs at ~$0.0008/page, ~25 % higher, almost entirely because
the spot price for a 24 GB card rose from ~$0.15 to ~$0.21-0.265/hr - GPU-hours per
page are comparable. In other words the pipeline did not get slower; the market
did.

## 7. Known open problems

1. **Local API route.** From the operator machine, `console.vast.ai:443` times out
   while everything else works. **This is not a blocker and must not be used as an
   excuse.** `cloud.vast.ai` serves the *same* API on the same paths: user object,
   v1 instance list, and teardown all answer normally (verified: status returned
   the balance, `vast down` reached the destroy path, `/api/v1/instances/` returned
   200). Point the tooling at it with `VAST_API_BASE=https://cloud.vast.ai`
   (`mth/vast.py` honours it for both the REST calls and the CLI subprocess), and
   set it in the user environment so the scheduled guard uses it too. Only if
   *both* hosts are unreachable is the API genuinely down - and then the answer is
   to run the call from the rented machine over ssh, not to give up on
   supervision.
2. **34.8 MP pages.** Need a 45-48 GB card, or an image that downscales for LaMa
   and composites back.
3. **Clean-page detection** remains manual per archive (`skip_pages`).
4. **Seven archives left**, ~2,100 pages, including a 381-page one.

## 7.5 How to tell whether the batch is actually working

The question came up repeatedly during the run and was answered wrongly twice.
Use only these signals:

| signal | healthy | what it means |
|---|---|---|
| GPU utilisation | 30-99 % while translating; short 0 % dips between pages | the model is generating |
| GPU power draw | near the model's ceiling (~280 W on a 3090, ~118 W idle) | real matrix work, not memory shuffling |
| newest file mtime in the work dir | within the last minute during OCR | pages are being detected and erased |
| run log | a new `... -> verified` line per archive | an archive finished and passed verification |
| heartbeat | refreshed within the last minute | the runner loop is alive |

Do **not** use: process count (the venv shim doubles every script), a chat latency
probe during a batch (it queues behind the in-flight request and measures queue
time), the console's "Mbps" badge (advertised bandwidth, not traffic), or "it has
been a while" (a 255-page archive legitimately takes ~70 minutes).

## 8. What an arriving agent should not do
* Do not rent before checking `deploy/VASTAI_PLAYBOOK.md` §3 and the current
  balance.
* Do not kill a `loading` host for being slow; it is free.
* Do not judge "is it working" from process counts or latency probes: look at GPU
  utilisation, GPU power draw, and file write timestamps.
* Do not change pipeline or config while a batch is running.
* Do not destroy a machine, or delete a delivered archive, without explicit
  confirmation.
* Do not re-run an archive just because it looks slow; `run_list.py` skips
  anything that verifies, so delete only on a verification failure.

## 9. Resume procedure

```powershell
python hanhua.py vast up --yes --image <repo>:0.5.0 --baked `
    --gpu "RTX 3090" --min-down 200 --max-price 0.28 --attempts 12 `
    --write-config config.json
python tools/gate.py --config config.json
python tools/run_list.py --manifest batch.json --config config.json `
    --outdir <deliveries> --log run.jsonl `
    --reset-gpu --ssh-host <host> --ssh-port <port> --ssh-key vast_ssh
```

`run_list.py` touches a heartbeat every 30 s while it works; the scheduled
`HanhuaGuard` destroys the instance when that heartbeat goes stale. When the run
finishes it destroys the machine itself (only after every archive verifies).
After the batch: revoke the API key, and remove it from the user environment if it
was stored there for the scheduled task.

## 10. The one-paragraph version

The pipeline was sound; every problem was in the material (text printed on art,
duplicate clean pages, huge pages) or in the operating discipline (not measuring
before acting, no supervisor that outlives the session). The fixes that mattered
were small and verified offline: collect every page extension, copy textless
pages, merge duplicate boxes, cap box area, release VRAM before inpainting. The
money that was wasted went to judgement errors under time pressure, not to the
translation itself - and the only durable defence was making the machine's
lifetime independent of whether anyone was watching.
