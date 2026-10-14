# llm_handoff/ — index

Handoff documentation for the **hanhua** project. Everything an engineer or AI
agent needs to continue seamlessly from the current state.

> All documents are in English. Nothing here contains secrets, personal data, IP
> addresses, or any content from the input material.

## Read in this order

| # | File | What it covers |
|---|---|---|
| 0 | `INDEX.md` | this index |
| 1 | `01_overview.md` | motivation, goals, non-goals, what works today |
| 2 | `02_architecture.md` | pipeline stages, components, data flow, JSON schema |
| 3 | `03_cli_and_config.md` | every command and every config field |
| 4 | `04_gpu_deployment.md` | vast.ai automation, the Docker image, readiness detection |
| 5 | `05_machine_reports.md` | anonymised history of every rented machine and its outcome |
| 6 | `06_bugs_and_fixes.md` | every bug hit and how it was fixed |
| 7 | `07_code_review.md` | strict review findings and remaining issues |
| 8 | `08_quality_workflow.md` | term/name consistency, the translation prompt |
| 9 | `09_testing_and_verification.md` | selftest, dry runs, delivery verification |
| 10 | `10_current_state_and_next_steps.md` | exact state + prioritised next steps |
| 11 | `11_lessons_and_guardrails.md` | distilled cost-safety rules |
| 12 | `12_hf_network_audit.md` | every function that can reach the network, and every HuggingFace touchpoint |
| 13 | `13_operating_manual.md` | **the runbook**: measured cost model, the exact procedure, what made it work, verification, and a failure-lookup table |
| 14 | `14_thrashing_postmortem.md` | **read before changing anything**: how this project repeatedly broke its own fixes, every regression, the anti-patterns, and the loop that works instead |
| 15 | `15_run_log.md` | **the full session, attempt by attempt**: every wrong diagnosis and its cost, the batch, the model upgrade, the detection over-call, the infrastructure failures, and the final state |

## Related documents (kept in place, not duplicated here)

| File | Purpose |
|---|---|
| `README.md` | project overview for users |
| `AGENTS.md` | operating instructions for AI agents |
| `deploy/VASTAI_PLAYBOOK.md` | hard rules + pitfall table for GPU rental (read this first if you touch vast.ai) |
| `llm_handoff/13_operating_manual.md` | the verified runbook + measured cost model |
| `deploy/CODE_REVIEW.md` | review findings (mirror of doc 7) |
| `deploy/Dockerfile` | the prebuilt GPU image |
| `deploy/vast_watchdog.py` | enforces "one machine at a time" while a run is in flight |
| `skills/vastai-cost-safety/` | portable cost-safety skill |

## TL;DR of the current state

- The **core pipeline works end to end**: archive in → translated archive out,
  same page count and order.
- The **GPU stack is packaged** in `deploy/Dockerfile` (not yet built/verified
  because no Docker was available on the authoring machine).
- The **vast.ai automation** now uses a prebuilt image + HTTP readiness polling
  (no SSH required for the happy path).
- Known open items are listed in `10_current_state_and_next_steps.md`.
