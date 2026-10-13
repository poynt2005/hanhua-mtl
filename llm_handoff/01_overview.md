# 01 — Overview

## Motivation

Translating a comic/manga by hand is a pipeline of six manual steps: sourcing
images, translating, proofreading, cleaning the artwork, typesetting, and
packaging. Each step has a mature open-source tool, but they are separate and the
whole thing is tedious and error-prone (especially keeping character names and
terms consistent).

This project automates the whole pipeline while **keeping every step
inspectable**: an intermediate JSON holds the detected text boxes and their
translations, so any step can be re-run or hand-corrected without redoing the
rest.

## Goals

1. **One archive in, one archive out.** Give it a `.zip` / `.cbz` / `.epub`; get
   back a translated archive with the same format, the same page names, and the
   same order.
2. **Delegate, do not reinvent.** Use existing tools for detection, OCR,
   inpainting, translation, and simplified→traditional conversion.
3. **Model-based quality.** Erase text with a neural inpainting model (not
   colour heuristics), read text with a manga-specialised OCR model, and
   translate with a model tuned for this kind of dialogue.
4. **Consistency is a feature.** One source term maps to exactly one canonical
   translation, enforced by a glossary and verified before delivery.
5. **Cheap and safe to run.** The GPU work is packaged so a rented box boots
   ready quickly, and the tooling enforces cost-safety rules.

## Non-goals

- It is not a human-quality translation; output is **machine translation** and
  should be labelled as such.
- It does not acquire source images.
- It does not guarantee pixel-perfect layout; long vertical lines may need manual
  adjustment.

## What works today

- `hanhua.py auto -i <archive>`: unpack → OCR(+erase) → translate → consistency
  check → typeset → repack, producing `<stem>_MTL_zhTW<ext>` plus a
  `.project.json`.
- Multiple OCR backends: a self-hosted HTTP service, a local model, OpenCV
  heuristics, a sidecar file, or a mock for offline testing.
- Two translation profiles: any OpenAI-compatible LLM, or a self-hosted
  visual-novel-tuned model.
- Term extraction, glossary review, normalisation, and verification.
- A prebuilt GPU image definition and a `vast` command to rent/deploy/destroy.

## Status at a glance

| Area | State |
|---|---|
| Core pipeline | working, verified on a sample archive |
| Local (CPU) path | works (slower); inpainting limited to lightweight methods |
| GPU stack | defined in `deploy/Dockerfile`; **not yet built/verified** |
| vast.ai automation | works; happy path no longer needs SSH |
| Cost safety | rules + playbook + skill in place |
| Open items | see `10_current_state_and_next_steps.md` |
