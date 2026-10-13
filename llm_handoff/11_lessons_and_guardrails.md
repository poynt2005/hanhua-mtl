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
   minutes, not tens of minutes.
4. **Readiness means "the model is present", not "the port answers".** A server
   with no model still returns 200.
5. **Do not require unrelated services in a health check.** One rate-limiting
   site made almost every machine look broken.
6. **One controller at a time.** Destroy leftovers before starting; a killed run
   can leave a machine billing.
7. **Always destroy on error.** Wrap the rent loop so an exception cannot leak a
   machine.
8. **Do not lose work on failure.** Keep the work directory; a failed run should
   not delete already-cleaned pages.
9. **Keep the machine until the output is confirmed.** Re-renting for every fix
   is the expensive path.
10. **Judge by content, not line counts.** Dead servers append blanks.
11. **Never upload weights; download on the remote.**
12. **Keys live in the environment only**, and are revoked when done.

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
