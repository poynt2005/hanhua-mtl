# 17 — Run report: the second batch, as actually executed

This is the record of what happened when `16_batch_operations.md` was executed.
It is deliberately written after the fact and contains **no input titles, hosts or
keys** - only counts, timings and failure modes, so it can live in a public repo.

## 1. Outcome

| | |
|---|---|
| archives to translate | 39 (4 more were already Chinese and excluded) |
| **delivered and verified** | **38** |
| **pages delivered** | **6,255** (5.87 GB) |
| not delivered | 1 archive: its pages are 34.8 MP and inpainting needs 25.45 GiB on a 24 GB card |
| machines used | several; every teardown confirmed with the provider API |
| spend | ~$10.6 (of which ~$3.2 was operator error, see §5) |

The run finished on 2026-10-23: `38/38 archives verified`, machine destroyed
automatically. A final aggregate pass re-checked every delivery against its source
(page count, page names, page order, image dimensions): 38 passed, 0 failed.

Every delivered archive was verified the same way: same page names, same order,
unchanged dimensions, and 100 % of detected blocks translated. A representative
line:

```
VERIFY  in=255 out=255 order=True dim=ok text_pages=195 blocks=2105
        translated=2105 (100.0%) -> OK
255 pages in 4170s (16.4 s/page) -> verified
```

## 2. Measured throughput

Per archive, wall clock from unpack to verified zip, including one model reload:

| pages | seconds | s/page | note |
|---|---|---|---|
| 79 | 990 | 12.5 | |
| 87 | 1,530 | 17.6 | |
| 100 | 990 | 9.9 | |
| 121 | 1,170 | 9.7 | |
| 134 | 1,831 | 13.7 | mixed extension archive |
| 143 | 2,130 | 14.9 | |
| 153 | 1,710 | 11.2 | |
| 161 | 3,120 | 19.4 | re-run after a misdiagnosed stall |
| 165 | 2,760 | 16.7 | |
| 192 | 2,460 | 12.8 | |
| 195 | 3,090 | 15.9 | |
| 218 | 2,520 | 11.6 | |
| 242 | 2,760 | 11.4 | |
| 244 | 3,360 | 13.8 | |
| 255 | 4,170 | 16.4 | |

**Median ≈ 13 s/page; the original estimate was 15 s/page.** Throughput therefore
met the plan. The spread is driven by how much text a book carries, not by the
pipeline: a CG set with eight text boxes per page costs roughly twice a book with
three.

## 3. What the money actually bought

| item | share |
|---|---|
| translation + OCR + inpainting | ~70 % |
| model reload between archives (5-6 min each) | ~15 % |
| the mistakes in §5 | ~15 % |

Marginal cost per page in steady state: **≈ $0.0008** (13 s at $0.22-0.265/hr).
That is roughly a quarter of the session-average cost, which is what a learning
run looks like: the early pages paid for the fixes.

## 4. Failure modes met, in order

1. **Offer churn.** `create instance` refused 8-10 offers in a row with
   `no_such_ask` while the search list still showed them. Re-querying and retrying
   is the only fix; it is not "no GPUs available".
2. **A `loading` host was destroyed as "slow"** - twice. Pulling is not billed, so
   this saved nothing and threw away the download.
3. **The OCR service's VRAM grows without bound** (0.4 GB at boot, 17 GB after 38
   large pages) and is never returned. With a 14B model resident the card is
   oversubscribed and a translation request waits forever. Fixed by stopping the
   OCR service after extraction and restarting it before the next archive.
4. **`pkill -f ocr_server.py` kills its own shell** and the restart loop with it,
   because the command line of the loop (and of the ssh command that sends the
   kill) also contains that string. Fixed by running the loop from its own script
   and killing only the python child with `pkill -x -f`.
5. **A 34.8 MP page cannot be inpainted on a 24 GB card** - LaMa asks for 25.45 GiB
   and the service silently degrades to blurring the text. That archive is out of
   scope for this card.
6. **The local runner is a single point of failure.** It was killed twice by the
   environment restarting; once the machine then billed for six hours unattended.
   Fixed with a heartbeat plus a scheduled guard that destroys a stale machine.

   That six-hour gap deserves its own note, because the fix is not obvious from
   the symptom. At the moment the runner died there was **no supervisor of any
   kind**: the scheduled guard did not exist yet (it was written the next
   morning), the monitoring subagent did not exist yet either, and the operator
   was waiting for a "background command finished" notification - which a killed
   process never sends, because it did not finish. Nothing was watching, so
   nothing could notice; the machine was found six hours later only because the
   user asked for a status report. The lesson is not "watch more carefully": an
   operator, however attentive, is not a supervisor. Supervision has to be a
   process that runs whether or not anyone is looking, and it has to be in place
   *before* the batch starts.
7. **Two "duplicate" `run_list` processes were not duplicates** - the venv's
   `python.exe` is a shim that re-execs the real interpreter, so every script
   appears twice in the process list. Acting on that observation killed a healthy
   run.
8. **The operator's route to `console.vast.ai` died** while the machine stayed
   reachable. Balance checks and teardown failed locally. Workaround: run the API
   call from the rented machine over ssh (`curl` with the key), which works
   because the machine has normal egress.

## 5. Operator errors and what they cost

| error | cost |
|---|---|
| killed a `loading` host twice (free pull thrown away) | time only |
| "we cannot see pull progress" - it was in `status_msg` all along | one wasted round of analysis |
| `pkill` self-match, twice | two archives re-run |
| "only unload the model for big pages" - broke inpainting for normal pages | 2 archives re-run |
| declared a stall while the GPU was at 98 %, killed a healthy run | ~40 min |
| destroyed a machine the user had just said to keep | one machine |
| mistook the venv shim for a duplicate runner, killed the batch | ~12 min |

None of these was a code defect in the pipeline; all were judgement failures
under time pressure. The countermeasure that actually helped was mechanical, not
attentional: measure before acting (GPU %, file writes), and let something other
than the operator decide when money stops.

## 5.1 The last day: the link, not the GPU, was the bottleneck

The remaining archives ran at 20-50 s/page while the GPU sat at 0 % and the host
CPU was 97 % idle. A direct measurement of one `/prepare` round trip explained it:
1.12 MB up plus 1.75 MB down took 49.8 s, an effective **0.19 Mbps**. A 20 MB
download did not finish in 300 s. The same code on an earlier host had run at
4.6-8 s/page.

Countermeasure: rent, then **measure the link before running anything** - pull
10 MB from the host over ssh and require at least 1 MB/s. With that gate the next
host measured 2.09 MB/s and the same archive class ran at **5-7 s/page**, a 6x
improvement with no code change. Two later hosts measured 1.09 and 1.42 MB/s and
were accepted; both behaved as expected.

Two smaller defects surfaced in the same session:

* the clean-page marker was still not enough on its own. One archive names its
  whole second half `..._n01_1` ... `..._n26_6` with **no `_t` counterpart**, so
  pairing never fired and 55 of 138 clean pages were translated. A page carrying
  the clean marker is now cleared whether or not a twin exists (109/109 and
  138/138 byte-identical afterwards).
* a JSON manifest written by PowerShell carried a UTF-8 BOM, and `json.load`
  refused it; the runner died immediately and the machine idled for six minutes.
  Manifests are now read with `utf-8-sig`.

Final state: **38 archives / 6,255 pages verified**, machine destroyed by the
runner itself, `remaining instances: []`.

## 6. How to resume

```powershell
# 1. rent (image is cached on hosts that ran it before)
python hanhua.py vast up --yes --image <repo>:0.5.0 --baked `
    --gpu "RTX 3090" --min-down 200 --max-price 0.28 --attempts 12 `
    --write-config config.json

# 2. measure the link BEFORE spending anything on it: pull 10 MB from the host
#    over ssh. Below ~1 MB/s every page costs 30-50 s, whatever the GPU does.
#    (A host at 0.07 MB/s looked exactly like "a slow model".)

# 3. gate
python tools/gate.py --config config.json

# 4. run the manifest: completed archives are skipped automatically
python tools/run_list.py --manifest batch.json --config config.json `
    --outdir <deliveries> --log run.jsonl `
    --reset-gpu --ssh-host <host> --ssh-port <port> --ssh-key vast_ssh
```

`run_list.py` verifies an existing output before skipping it, so a half-finished
batch resumes exactly where it stopped. Do **not** delete a partially written
archive to force a re-run unless it fails verification - the verifier is the
authority, not the file size.

## 7. What is deliberately not recorded here

Input titles, page contents, host addresses, instance ids and keys. The per-archive
timings live in `run_batch*.jsonl` next to the deliveries, which is local-only by
design.
