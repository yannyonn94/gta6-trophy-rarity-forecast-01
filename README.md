# GTA VI Trophy Rarity Forecast

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate
on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%,
75%, 95%) per trophy. Entry for
[`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun. Scored by mean pinball loss on the logit scale; lower is
better, and a prize needs a loss strictly below the best house entry.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in`
in the input directory (default `/input`) and writes `<id>.out` for each
to the output directory (default `/output`). Deterministic: two runs
write identical bytes.

## The model

A trophy's rate is **earners / players tracked**, and Exophase refreshes
its counts in steps hours apart: when a refresh lands,
`players_tracked` jumps and every rate falls, because the newly tracked
players hold far fewer trophies. A rate-only trend confuses that
dilution with players un-earning trophies, so the program fits two
trends per trophy against log-time `x = ln(1 + hours since
2026-11-19T00:00Z)`:

- `log(earners)`, where `earners = rate * players_tracked / 100`
- `log(players_tracked)`

and reads the median rate at the target as `earners_hat / players_hat`.
Earners are monotone, so the earners forecast is floored at the last
observed earner count and capped at the forecast player count. The
median is blended 50/50 in logit space with a direct least-squares
trend of `logit(rate)` on the same axis. Slopes are shrunk toward zero
(Wiener factor from the slope's standard error, times `m/(m+2)` for `m`
distinct snapshots) and capped, and the median logit may move at most
2.5 from the last observed rate, so a short noisy history cannot
extrapolate wildly.

History is collapsed to distinct snapshots first: Exophase holds a page
unchanged for hours, so most reads repeat the previous one exactly and
would otherwise let a long flat stretch outweigh the few reads where
anything moved.

The logit spread is built from four parts, added in quadrature:
a random walk (`0.45 * sqrt(days)`), the slope uncertainty of the two
fits times the extrapolation distance, the residual volatility of the
logit series, and refresh-step risk. Refresh risk is measured from the
history when refreshes were seen (how often `players_tracked` jumped,
and how far this trophy's rate moved at each jump). When the history is
perfectly flat — launch night is a single distinct snapshot — a prior
is used instead (about one refresh every 5 hours, each moving a rate by
roughly a logit unit) plus a small prior drift on the player trend:
players keep being tracked while newcomers hold trophies at a lower
rate, so a flat history still forecasts a gentle fall. A trophy the
last read shows at (essentially) 100% gets a tight band just under 100%:
it survived the launch-night refresh at 100%.

## Files

- `main.py` — the whole forecaster (single file, entrypoint)
- `README.md` — this file
- `LICENSE` — MIT

## Verification

Run on the kit's practice cases (`p00`, `p01`: checkpoints 03:00 and
03:30 UTC 19 Nov 2026, 90-minute horizons) and check the outputs are
valid JSON with 55 trophies, five non-decreasing quantiles each in
0–100, each file under 65,536 bytes, and byte-identical across two runs.

Measured 21 Nov 2026 (local run, Python 3.12): `p00.out` 2,794 bytes
and `p01.out` 2,768 bytes, both valid on every check above, and two
runs wrote byte-identical files (`diff -r` clean). Both practice
histories are a single flat snapshot (638 players tracked, every rate
unchanged), so the forecasts come from the flat-history prior: medians
drift gently down (e.g. trophy 1, last rate 0.63%, median 0.55%) with a
wide band (logit spread ≈ 0.56) that prices the refresh the practice
answers contain. The 100% trophy (id 2) is forecast at 99.97% with a
tight band, as described above.
