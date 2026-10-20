# 16 — Running a second batch (the manifest runbook)

This document is the handoff for a **second delivery batch**, run after the one
described in `15_run_log.md`. It records what is different about this batch, how
the cost was estimated, and which decisions were deliberately *not* taken.

Nothing here contains input material, filenames, hosts or keys.

## 1. What the batch is

| | |
|---|---|
| archives | 43 (single-volume `.zip`) |
| pages | 6,557 |
| already translated, excluded | 4 archives / 281 pages |
| **to translate** | **39 archives / 6,276 pages** |

The four excluded archives were checked by opening pages, not by trusting their
filenames: one was machine-translated Simplified Chinese, three were Traditional
Chinese. Excluding them was a user decision.

Page geometry is ordinary: **2.17 MP average**, comparable to the first batch, so
the measured 15 s/page baseline transfers. 131 pages are above 8 MP (the worst
archive averages 34.8 MP), which is where inpainting time will not be linear.

## 2. The structural finding: every archive ships its artwork twice

A large share of this batch is CG-style: the same drawing is published once with
dialogue printed on it and once clean. Three layouts were observed, and they all
pair up:

1. **explicit page markers** — `…_t00001` / `…_n00001` in the same archive;
2. **an appended clean half** — page *i* and page *i + N/2* are the same drawing;
3. **interleaved pairs** — text, clean, text, clean.

Measured across the batch: **26–50 % of pages carry no text at all**.

Two consequences:

- **They must not be translated.** They also must not be re-encoded: a
  decode/encode round trip only adds JPEG generation loss. `do_render` now copies
  such pages byte-for-byte (verified: 804,240 bytes in, 804,240 out).
- **They are a better source than inpainting.** For a page with text, if a twin
  exists, the translation can be typeset straight onto the clean twin — no LaMa,
  no erase artifacts. **Pairing is decided by filename, not by pixels.** An
  image-similarity rule was implemented and measured before shipping, and it
  failed: inside a single-scene CG set every clean page resembles every printed
  page (true pair MAD p50 18.3 vs best false candidate 19.2 on the 24-page set;
  10.7 vs 12.1 on the 221-page set). Such a rule would happily typeset one page's
  dialogue onto another page's drawing.

  `pair_by_names` therefore only accepts an explicit `…_t<n>` ↔ `…_n<n>` marker
  with the same number, plus identical pixel size. Measured on the one series
  that uses markers: **109/109 pairs, 0 wrong**. Every other archive keeps the
  normal erase path, because a miss costs nothing and a wrong pair is a silent
  quality defect.

This is the largest quality win available in this batch, and it costs no GPU
time.

## 3. Two pipeline fixes this batch required

Both were validated offline, with real pages, before any rental (no paid
experiment):

| Fix | Why it was required | Evidence |
|---|---|---|
| `list_images` now collects **every** page extension | `pattern` was a single glob, so an archive holding `.jpg` pages plus a `.webp` cover silently lost the webp page and then aborted repack with `translated page missing` | 3 of 43 archives affected; after the fix 21/21 and 134/134 pages round-trip with identical names, order and dimensions |
| textless pages are **copied, not rendered** | without it, ~26–50 % of pages were decoded and re-encoded for nothing | 12 of 24 pages byte-identical to the source, confirmed by hash |
| duplicate boxes for one column are **merged** | on text printed straight onto the art the detector returns a tight box and a loose box for the same column; translating and drawing both put two sentences on top of each other and the page became an unreadable smear | 91 blocks → 59 after merging on one 24-page archive; the re-render is readable and each column carries one continuous sentence |
| the translation model is **unloaded before OCR** | a 14B Q6_K holds 13.8 GB of a 23.6 GB card, leaving too little for LaMa | remote VRAM dropped from 22.9 GB to 3.0 GB during the detection pass |

## 4. Cost model

Measured inputs: 15 s/page for a text page on the baked Q6_K image (first batch),
~1.5 s/page for a page the detector finds empty.

| clean-page share | text pages | GPU-hours | cost at $0.19–0.23/hr |
|---|---|---|---|
| 0 % | 6,276 | 28.3 | $5.4–6.5 |
| 26 % (measured lower bound) | 4,644 | 22.2 | $4.2–5.1 |
| 40 % | 3,765 | 18.9 | $3.6–4.3 |
| 50 % (typical CG set) | 3,138 | 16.6 | $3.2–3.8 |

Plus ~2 h for the 131 oversized pages. Boot/pull adds nothing: **image pull is
not billed** (V69), only storage.

The exact clean share is only known after the detector runs, which is why the
first two archives are small ones.

## 5. Monitoring without SSH

The renter API has no per-instance traffic counter. Everything below goes through
the API, so a missing sshd during the pull is not a signal of anything (V71):

| Question | How |
|---|---|
| how far is the image pull? | per-layer events in `status_msg` (`tools/pull_watch.py`) |
| is the box alive? | `vastai logs <id>` |
| GPU % / VRAM / power / rx-tx | `python tools/monitor.py --instance <id>` (`/proc/net/dev` + `nvidia-smi` via `vastai execute`) |
| why did it fail to boot? | `vastai logs` **before** destroying anything |

## 6. Decisions taken, and deliberately not taken

Taken:

- **manifest-driven runner** (`tools/run_list.py`) — `run_batch.py` identifies a
  volume by page count, and this batch contains two archives of 242 pages.
  The manifest also makes the batch resumable and records seconds/page.
- **one machine, on-demand** for the first run: the cheapest measured path.
- **dead-man switch** for unattended runs (R12), `stop` not `destroy`.

Not taken, on purpose:

- **concurrency / batching in the translate stage.** It is the only lever that
  lowers GPU-hours (an LLM at batch 1 leaves most of the card idle), but it
  multiplies the failure modes this project already paid for: truncated context,
  model reloads, translations landing on the wrong page. Machines are cheaper to
  add than bugs (R13).
- **spot instances.** Cheaper per hour than on-demand, but unproven here;
  interruption costs a re-run of the archive in flight.
- **re-quantising the model.** Quality must stay comparable to batch 1.

## 7. Open risks

1. **131 oversized pages** — inpainting cost is not linear in pixels; one archive
   averages 34.8 MP and may be slow or hit VRAM limits. It is small (21 pages),
   which makes it a cheap early test.
2. **Text printed directly on artwork** (no speech bubbles) in the CG sets. This
   is harder to erase cleanly than manga bubbles; the clean-twin path avoids it
   wherever a twin exists.
3. **Two archives share a page count** — never identify an archive by its length.
4. **Term consistency across a series** — several archives belong to one series.
   A per-series glossary reviewed once is cheaper than reviewing 39 books.
