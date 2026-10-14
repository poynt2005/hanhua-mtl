# 11 — Lessons and guardrails

Distilled from the development experience. The operational version lives in
`deploy/VASTAI_PLAYBOOK.md`; the portable version is the skill in
`skills/vastai-cost-safety/`.

## The one that mattered most

> Almost all of the money was **not** spent on the model — it was spent running
> unvalidated setup scripts for the first time on a paid machine, and on
> misreading a client-side failure as a machine failure.

Two concrete forms:

1. **A private-key permission problem** made every `ssh`/`scp` fail silently on
   Windows. The network check returned empty output, which looked like a bad
   network, so good machines were destroyed one after another.
2. **A fragile multi-dependency install** (torch/transformers/OpenCV system
   libs/a model server) was re-run from scratch on every boot, and every missing
   piece surfaced only on a paid machine.

## Guardrails that follow from this

1. **Bake the environment into an image and smoke-test it at build time.** The
   build is free; the debugging on a paid box is not.
2. **Measure on the machine; do not trust advertised figures.** Offer bandwidth
   and reliability did not predict success.
3. **Fail fast on a bad download path.** Probe the speed at boot and give up in
   minutes, not tens of minutes. Docker Hub throughput is **per host**: one host
   pulled the 15.5 GB image at 3.6 MB/s (~60 min), another with warm layers was
   `running` in **187 s**.
4. **Readiness means "the model is present", not "the port answers".** A server
   with no model still returns 200.
5. **Do not require unrelated services in a health check.** One rate-limiting
   site made almost every machine look broken.
6. **One machine at a time is a hard invariant.** `vast up` refuses while one of
   ours is running or loading; the owned instance is recorded; a watchdog
   destroys extras. **Never write ad-hoc rental code** - that is exactly how a
   second machine started while the first was still billing.
7. **Never send a long silent request.** A non-streaming completion is silent
   while the model generates and the host's port forward drops the idle socket
   after ~15 s (measured 4/6 vs 6/6 with streaming). Stream it, and retry
   transient transport errors in **both** clients (LLM and OCR).
8. **Build in diagnosability before renting**: `faulthandler` to a file,
   per-stage markers, VRAM in `/health`, a `GET /logs` endpoint, a restart loop.
   A silent native crash wastes an entire rental.
9. **Keep the machine when a run fails** - that is when you need it. A
   `finally: destroy` guard once threw away the only debugging environment.
10. **Do not lose work on failure.** Keep the work directory.
11. **Judge by content, not line counts.** Dead servers append blanks; duplicated
    log lines can make a progress counter go *down*.
12. **Identify work items unambiguously.** Two different volumes had 237 pages;
    page count alone is not an identity.
13. **Never upload weights; download on the remote.**
14. **Keys live in the environment only**, and are revoked when done.

## Money rules (short form)

- Report the budget and get consent before renting.
- $1 cap per task; report at 80%; stop at 100%.
- The first line of every report is the money.
- Destroy and verify; never leave a machine unattended.

## Quality rules (short form)

- Erase with a model and a **text mask**, not whole rectangles.
- Preserve image quality on re-encode (source quality, no chroma subsampling).
- Translate with the model's tuned prompt, recent context, and small batches.
- One source term → one canonical translation; verify before delivery.
- Label output as machine translation.

## Honest retrospective

- The correct approach (models + a baked image) was identified early but the
  automation was rewritten and shipped without validating each piece, and the
  pieces were then debugged on paid machines. That ordering was the mistake.
- The pipeline itself was sound from the start; the failures were in deployment
  and process, not in the translation logic.
- **The single most expensive misdiagnosis**: an aborted batch looked like an OCR
  server crash (`Remote end closed connection`), so the detector was suspected
  and rebuilt. The real cause was that a **non-streaming LLM request is silent
  while the model generates**, and the host's port forward drops an idle socket.
  The stage markers proved `/prepare` had been returning 200 the whole time.
  Lesson: instrument first, guess second - and stream long requests.
- **The second most expensive mistake** was writing a throwaway script that could
  rent machines. When a slow host was destroyed by hand, that script read
  "instance vanished" as "try the next offer" and rented a second machine while
  the first was still billing. Rental now happens through exactly one guarded
  path, with a watchdog enforcing one instance.

## What it costs when it goes right

One V100 ($0.153/hr), five volumes (~1,850 pages), each verified, **under $1
total** - about **15 cents per 250-page volume**. That is the number to compare
against: anything much higher means idle time, a slow host, or a stall, and
should be investigated immediately rather than absorbed.
