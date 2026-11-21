"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate
on Exophase at the case target, as five quantiles (5/25/50/75/95%).

Model: earners over players, not rates alone.
--------------------------------------------
A trophy's rate is earners / players_tracked. Exophase refreshes its
counts in steps, hours apart: when a refresh lands, players_tracked jumps
and every rate falls, because the newly tracked players hold far fewer
trophies. A rate-only trend fit confuses that dilution with players
un-earning trophies. So this program fits two trends per trophy, both
against log-time x = ln(1 + hours since the 2026-11-19T00:00Z release):

  * log(earners)     -- earners = rate * players_tracked / 100
  * log(players)     -- the tracked-player count itself

and reads the median rate at the target as earners_hat / players_hat.
Earners are monotone (players cannot un-earn a trophy), so the earners
forecast is floored at the last observed earner count and capped at the
forecast player count. The median is blended 50/50 in logit space with a
direct least-squares trend of logit(rate) on the same log-time axis, which
guards the decomposition when rates are large and the log approximation
is loose. Slopes are shrunk toward zero (a Wiener-style factor from the
slope's own standard error, times m/(m+2) for m distinct snapshots) and
capped, so a short noisy history cannot extrapolate wildly; the median
logit is also capped to a move of 2.5 from the last observed rate.

Spread: the logit spread is honest about the three things that can move
a rate -- a random walk (0.45 * sqrt(days)), the slope uncertainty of the
two fits times the extrapolation distance, the residual volatility of
the logit series, and refresh-step risk. Refresh risk is measured from
the history itself when refreshes were seen (how often players_tracked
jumped, and how far this trophy's rate moved at each jump); when the
history is perfectly flat (launch night: one distinct snapshot), a prior
is used instead -- about one refresh every 5 hours, each moving a rate
by roughly a logit unit -- and a small prior drift is applied to the
player trend (players keep being tracked; new players hold trophies at
a lower rate, so a flat history still forecasts a gentle fall).

Deterministic: no randomness anywhere; cases are read in name order and
every number is rounded half-up to 4 decimals, so two runs write
identical bytes. Standard library only, no network.

Usage (outside the grader):
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs `python3 -P -s /bot/main.py` with MAKEFUN_INPUT_DIR /
MAKEFUN_OUTPUT_DIR unset, so /input and /output are the defaults.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Rates are clamped to this band before any logit, as the scorer does.
LOW_PCT = 0.01
HIGH_PCT = 99.99
# A trophy the last read shows at (essentially) 100% gets a tight band
# just under 100 instead of the generic spread: it survived the launch
# night refresh at 100%, so the honest call is "still ~100%".
FULL_PCT = 99.9
FULL_MEDIAN_PCT = 99.97
FULL_SIGMA = 0.35

# Random-walk volatility of a logit rate, per square-root day.
WALK_SIGMA = 0.45
# Prior refresh cadence (one per 5 h) and per-refresh logit move, used
# only when the history shows no refresh at all.
PRIOR_REFRESH_HOURS = 5.0
PRIOR_JUMP_LOGIT = 1.0
# Prior player-trend slope (log players per log-time unit) for a flat
# history, and the share of it the earner trend inherits: newcomers
# hold trophies at a lower rate than the players already tracked.
PRIOR_PLAYER_SLOPE = 0.45
PRIOR_EARNER_SHARE = 0.30

MAX_MEDIAN_MOVE = 2.5   # cap on |median logit - last logit|
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime ('Z' handled)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(moment):
    """x axis: ln(1 + hours since release), floored at the release."""
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log1p(max(hours, 0.0))


def clamp_pct(pct):
    return min(HIGH_PCT, max(LOW_PCT, pct))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(logit_value):
    if logit_value >= 0:
        p = 1.0 / (1.0 + math.exp(-logit_value))
    else:
        e = math.exp(logit_value)
        p = e / (1.0 + e)
    return clamp_pct(p * 100.0)


def round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def least_squares(xs, ys):
    """Ordinary least squares y ~ a + b*x.

    Returns (intercept, slope, slope_standard_error, residual_std).
    With fewer than 2 points the slope is 0 and the intercept the mean.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    if n == 1:
        return mean_y, 0.0, 0.0, 0.0
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    if n > 2:
        rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid_var = rss / (n - 2)
        slope_se = math.sqrt(resid_var / sxx)
        resid_std = math.sqrt(resid_var)
    else:
        slope_se = 0.0
        resid_std = 0.0
    return intercept, slope, slope_se, resid_std


def shrink_slope(slope, slope_se, n_points):
    """Shrink a fitted slope toward zero.

    Wiener factor slope^2 / (slope^2 + se^2) when the standard error is
    known, times n/(n+2) so a handful of snapshots cannot dominate.
    """
    if slope == 0.0:
        return 0.0
    factor = n_points / (n_points + 2.0)
    if slope_se > 0.0:
        factor *= (slope * slope) / (slope * slope + slope_se * slope_se)
    return slope * factor


def clamp_value(value, low, high):
    return min(high, max(low, value))


def distinct_snapshots(history, trophy_count):
    """Collapse the history to its distinct states.

    Exophase holds a page unchanged for hours, so most reads repeat the
    previous one exactly. Keeping every read would let a long flat
    stretch outweigh the few reads where anything moved. A state is
    distinct when the player count or any rate differs from the state
    before it; the first read of each state is kept, oldest first.
    """
    states = []
    previous = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == previous:
            # Same state as the read before it: keep the freshest read
            # of the state, so the state's timestamp is the last time
            # it was seen holding, not the first.
            states[-1] = read
            continue
        previous = key
        states.append(read)
    return states


def forecast_trophy(rates_by_state, players_by_state, xs, x_target,
                    horizon_days, horizon_hours, last_rate):
    """One trophy's five quantiles at the target.

    rates_by_state / players_by_state / xs are parallel lists over the
    distinct snapshots, oldest first; a None rate means that snapshot
    lacked this trophy.
    """
    points = [(x, rate, n) for x, rate, n in zip(xs, rates_by_state, players_by_state)
              if rate is not None and n]
    if not points:
        # No observation of this trophy at all: wide, centred low.
        median_logit = logit_of(0.5)
        sigma = 1.5
    else:
        last_x = points[-1][0]
        last_earners = max((points[-1][1] * points[-1][2]) / 100.0, 0.5)
        m = len(points)
        log_e = [math.log(max((r * n) / 100.0, 0.5)) for _, r, n in points]
        log_n = [math.log(n) for _, _, n in points]
        log_l = [logit_of(r) for _, r, _ in points]
        xs_p = [x for x, _, _ in points]

        a_e, b_e, se_e, _ = least_squares(xs_p, log_e)
        a_n, b_n, se_n, _ = least_squares(xs_p, log_n)
        a_l, b_l, se_l, resid_l = least_squares(xs_p, log_l)

        if m <= 1:
            # Flat history: no trend is estimable. Use the prior -- the
            # player base keeps growing, earners grow only a share of
            # that, so the rate drifts gently down.
            b_n_use = PRIOR_PLAYER_SLOPE
            b_e_use = PRIOR_EARNER_SHARE * PRIOR_PLAYER_SLOPE
            # Anchor the prior at the last observed state: the drift
            # applies over the forecast horizon only, never over hours
            # the history already shows as flat.
            a_n_use = log_n[-1] - b_n_use * xs_p[-1]
            a_e_use = log_e[-1] - b_e_use * xs_p[-1]
            logit_trend = None
            se_e = se_n = se_l = 0.0
            resid_l = 0.0
            jump_std = PRIOR_JUMP_LOGIT
            expected_jumps = horizon_hours / PRIOR_REFRESH_HOURS
        else:
            b_e_use = clamp_value(shrink_slope(b_e, se_e, m), -0.5, 3.0)
            b_n_use = clamp_value(shrink_slope(b_n, se_n, m), 0.0, 2.0)
            a_n_use, a_e_use = a_n, a_e
            b_l_use = clamp_value(shrink_slope(b_l, se_l, m), -2.5, 2.5)
            logit_trend = a_l + b_l_use * x_target
            # Refresh risk measured from the history: how often the
            # player count jumped, and how far this trophy's rate moved
            # at each jump (consecutive distinct states).
            ratios = []
            for (x0, r0, _), (x1, r1, _) in zip(points, points[1:]):
                if r0 and r1:
                    ratios.append(abs(logit_of(r1) - logit_of(r0)))
            if len(ratios) >= 2:
                mean_ratio = sum(ratios) / len(ratios)
                jump_std = clamp_value(
                    math.sqrt(sum((v - mean_ratio) ** 2 for v in ratios) / len(ratios)),
                    0.10, 0.90)
                jump_std = max(jump_std, 0.30)
            elif ratios:
                jump_std = clamp_value(ratios[0], 0.10, 0.90)
            else:
                jump_std = PRIOR_JUMP_LOGIT
            jumps_seen = m - 1
            span_hours = _logtime_span_hours(points)
            if jumps_seen >= 2 and span_hours > 1.0:
                # Jump rate per hour, measured from the history's own
                # refreshes, times the forecast horizon.
                expected_jumps = jumps_seen * (horizon_hours / span_hours)
            else:
                expected_jumps = horizon_hours / PRIOR_REFRESH_HOURS
            expected_jumps = min(expected_jumps, 40.0)

        players_hat = math.exp(a_n_use + b_n_use * x_target)
        earners_hat = math.exp(a_e_use + b_e_use * x_target)
        earners_hat = min(max(earners_hat, last_earners), players_hat)
        rate_hat = clamp_pct(100.0 * earners_hat / players_hat)

        last_logit = logit_of(last_rate if last_rate is not None else rate_hat)
        median_logit = logit_of(rate_hat)
        if logit_trend is not None:
            median_logit = 0.5 * median_logit + 0.5 * logit_trend
        median_logit = clamp_value(
            median_logit, last_logit - MAX_MEDIAN_MOVE, last_logit + MAX_MEDIAN_MOVE)

        walk = WALK_SIGMA * math.sqrt(max(horizon_days, 0.0))
        dx = max(x_target - last_x, 0.0)
        slope_var = (se_e * se_e + se_n * se_n) * dx * dx
        slope_part = math.sqrt(min(slope_var, 1.0))
        resid_part = min(resid_l, 1.0)
        refresh_part = math.sqrt(expected_jumps) * jump_std
        refresh_part = min(refresh_part, 1.2)
        sigma = math.sqrt(walk * walk + slope_part * slope_part
                          + resid_part * resid_part + refresh_part * refresh_part)
        sigma = clamp_value(sigma, SIGMA_FLOOR, SIGMA_CAP)

    if last_rate is not None and last_rate >= FULL_PCT:
        median_logit = logit_of(FULL_MEDIAN_PCT)
        sigma = FULL_SIGMA

    quantiles = []
    previous = None
    for z in Z_SCORES:
        value = round4(inv_logit_pct(median_logit + z * sigma))
        if previous is not None and value < previous:
            value = previous
        quantiles.append(value)
        previous = value
    return quantiles


def _logtime_span_hours(points):
    """Hours between the first and last snapshot, from their x values.

    x = ln(1 + hours since release), so hours = exp(x) - 1.
    """
    return max(math.exp(points[-1][0]) - math.exp(points[0][0]), 0.0)


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        # No history at all: emit a wide, low forecast rather than crash.
        return {"trophies": [
            {"id": t["id"], "q": [0.05, 0.2, 0.5, 1.4, 4.0]} for t in trophies]}

    target = parse_time(case["target"])
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0
    x_target = log_time(target)

    states = distinct_snapshots(history, len(trophies))
    xs = [log_time(parse_time(s["at"])) for s in states]
    players = [s.get("players_tracked") or 0 for s in states]
    last_rates = history[-1].get("rates") or [None] * len(trophies)

    rows = []
    for i, trophy in enumerate(trophies):
        rates = [(s.get("rates") or [None] * len(trophies))[i] if i < len(s.get("rates") or []) else None
                 for s in states]
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(rates, players, xs, x_target,
                                 horizon_days, horizon_hours,
                                 last_rates[i] if i < len(last_rates) else None),
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
