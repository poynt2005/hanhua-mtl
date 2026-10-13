# Hard rules

> These are rules, not suggestions. Breaking any of them = stop, destroy, report.
> If a rule seems not to apply, ask the user; do not decide unilaterally.

## R1 Before renting

1. **No renting without explicit user consent.** Consent is per machine; re-renting needs a new ask.
2. **Report the budget first**: "N machines × $X/hr × Y hours ≈ $Z".
3. **Write and syntax-check scripts locally** (`bash -n`) before renting. Do not iterate on a paid machine.
4. Rent only network-qualified offers: reasonable `inet_down`/`inet_up`, reliability floor, `--direct`.

## R2 Within 5 minutes of boot

5. Run `scripts/net_check.sh` and time a 10MB scp before any setup.
6. `NET_BAD`, or 10MB scp > 30s, or only a proxy ssh for a large-file task → **destroy and switch**.
   Do not change proxies, do not fall back to CPU, do not retry setup.
7. Three consecutive bad machines → **stop and report**; do not rent a fourth.

## R3 While running

8. Three gates before a batch: alive → single-sentence test → no stale process.
9. Judge success by **content and marker files**, not line counts or exit codes.
10. `pgrep -af` before sending a new command.
11. GPU idle 15 minutes with no download → destroy.
12. Same error fixed twice without success → **stop and report**.
13. Conclusions need a reproducible test.

## R4 Budget

14. **$1 cap per task** (measured on the account spend); report at $0.8, stop at $1.
15. **First line of every report is money.**

## R5 Teardown

16. On completion, stop, interruption, handoff, or leaving: **destroy first**, confirm the instance list is empty and billing stopped.
17. Destroy, then verify.
18. **Pull and verify artifacts before destroying.**
19. Never leave a machine running unattended.

## R6 Access

20. An agent forbidden from touching the cloud must not rent, ssh, or destroy.
21. **API keys live only in environment variables; never write them to files.**

## Violations

If you break a rule: stop, destroy everything, confirm billing stopped, and report
which rule and how much it cost. Do not hide it.
