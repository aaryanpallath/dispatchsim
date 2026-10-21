"""All experiments behind the README.

    uv run python bench/experiments.py      # results/experiments.json and results/*.png

1. Policies across fleet sizes: greedy vs batched matching from scarce to plentiful supply.
2. The batching window: how long to collect requests before matching.
3. Matching speed: one batched round with H3 candidate search vs brute force over the whole fleet, from 2,000 to
   50,000 drivers, checking that both find an assignment of the same total ETA.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from dispatchsim.data import load_demand  # noqa: E402
from dispatchsim.sim import MAX_ETA, NO_EDGE, RES, DriverIndex, Travel, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
START, HOURS = "2016-03-09 17:00", 3  # a Wednesday evening peak in Manhattan
FLEETS = [3000, 4000, 5000, 6500]
WINDOWS = [0, 2, 5, 10, 20, 30, 60, 120]


def run(args: tuple[int, float]) -> dict:
    fleet, window = args
    d = load_demand(ROOT / "data", START, HOURS)
    t0 = time.perf_counter()
    s = simulate(d.requests, Travel(d.speed_kmh, d.detour), fleet, window).summary()
    return {**s, "window_s": window, "sim_seconds": round(time.perf_counter() - t0, 1)}


def matching_speed(fleets=(2000, 5000, 10000, 20000, 50000), riders=300, idle_share=0.2, reps=5) -> list[dict]:
    d = load_demand(ROOT / "data", START, HOURS)
    travel = Travel(d.speed_kmh, d.detour)
    k = travel.ring_limit()
    rng = np.random.default_rng(0)
    pts = d.requests[["plat", "plon"]].to_numpy()
    rows = []
    for fleet in fleets:
        idle = int(fleet * idle_share)
        ix = rng.choice(len(pts), idle + riders, replace=True)
        dl, rl = pts[ix[:idle]], pts[ix[idle:]]
        index = DriverIndex(dl[:, 0].copy(), dl[:, 1].copy())
        for i in range(idle):
            index.add(i, dl[i, 0], dl[i, 1])
        timings = {"h3": [], "brute": []}
        totals = {}
        for method in ("h3", "brute"):
            for _ in range(reps):
                t0 = time.perf_counter()
                if method == "brute":
                    cost = travel.eta(rl[:, 0][:, None], rl[:, 1][:, None], dl[:, 0][None, :], dl[:, 1][None, :])
                    cost = np.where(cost <= MAX_ETA, cost, NO_EDGE)
                    drivers = np.arange(idle)
                else:
                    cands = [np.array(index.near(a, b, k), dtype=int) for a, b in rl]
                    drivers = np.unique(np.concatenate(cands))
                    pos = {dd: i for i, dd in enumerate(drivers)}
                    cost = np.full((riders, len(drivers)), NO_EDGE)
                    for i, c in enumerate(cands):
                        e = travel.eta(rl[i, 0], rl[i, 1], dl[c, 0], dl[c, 1])
                        ok = e <= MAX_ETA
                        cost[i, [pos[x] for x in c[ok]]] = e[ok]
                r, c = linear_sum_assignment(cost)
                timings[method].append((time.perf_counter() - t0) * 1000)
                good = cost[r, c] < NO_EDGE
                totals[method] = (int(good.sum()), round(float(cost[r, c][good].sum()), 3))
        rows.append(
            {
                "fleet": fleet,
                "idle_drivers": idle,
                "riders": riders,
                "h3_ms": round(float(np.median(timings["h3"])), 1),
                "brute_force_ms": round(float(np.median(timings["brute"])), 1),
                "same_assignment_cost": totals["h3"] == totals["brute"],
                "matched": totals["h3"][0],
            }
        )
        print(rows[-1], flush=True)
    return rows


def plots(fleet_rows: list[dict], window_rows: list[dict], speed: list[dict]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for pol, style in (("greedy", "o-"), ("batched 10s", "s-"), ("batched 30s", "^-")):
        rs = [r for r in fleet_rows if r["policy"] == pol]
        ax[0].plot([r["fleet"] for r in rs], [r["completed_pct"] for r in rs], style, label=pol)
    ax[0].set(title="Rides completed by fleet size", xlabel="drivers", ylabel="% of requests served")
    w = sorted(window_rows, key=lambda r: r["window_s"])
    ax[1].plot([r["window_s"] for r in w], [r["wait_p50_s"] / 60 for r in w], "o-", label="median wait")
    ax[1].plot([r["window_s"] for r in w], [r["wait_p90_s"] / 60 for r in w], "s-", label="90th percentile wait")
    ax1b = ax[1].twinx()
    ax1b.plot(
        [r["window_s"] for r in w], [r["completed_pct"] for r in w], "^--", color="gray", label="% served (right)"
    )
    ax1b.set_ylabel("% served", color="gray")
    lines = ax[1].get_legend_handles_labels()
    extra = ax1b.get_legend_handles_labels()
    ax[1].legend(lines[0] + extra[0], lines[1] + extra[1], fontsize=8, loc="center right")
    ax[1].set(title="Batching window at 4,000 drivers", xlabel="window (s, 0 = greedy)", ylabel="wait (min)")
    ax[2].loglog([s["fleet"] for s in speed], [s["brute_force_ms"] for s in speed], "o-", label="brute force")
    ax[2].loglog([s["fleet"] for s in speed], [s["h3_ms"] for s in speed], "s-", label="H3 candidate search")
    ax[2].set(title="One matching round (300 riders)", xlabel="fleet size", ylabel="ms")
    ax[2].set_xticks([s["fleet"] for s in speed], [f"{s['fleet'] // 1000}k" for s in speed])
    ax[2].minorticks_off()
    for a in (ax[0], ax[2]):
        a.legend(fontsize=8, loc="best")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "experiments.png", dpi=130)


def main() -> None:
    import sys

    if "--plots-only" in sys.argv:  # redraw from saved results without re-running the simulations
        r = json.loads((ROOT / "results" / "experiments.json").read_text())
        return plots(r["policies_by_fleet"], r["batching_window"], r["matching_speed"])
    jobs = [(f, w) for f in FLEETS for w in (0, 10, 30)] + [(4000, w) for w in WINDOWS if w not in (0, 10, 30)]
    with ProcessPoolExecutor(3) as ex:
        out = list(ex.map(run, jobs))
    fleet_rows = [r for r in out if r["window_s"] in (0, 10, 30)]
    window_rows = [r for r in out if r["fleet"] == 4000]
    speed = matching_speed()
    d = load_demand(ROOT / "data", START, HOURS)
    result = {
        "data": {
            "period": f"{START}, {HOURS} h, Manhattan",
            "requests": len(d.requests),
            "speed_kmh": round(d.speed_kmh, 2),
            "detour": round(d.detour, 3),
            "h3_resolution": RES,
            "h3_edge_km": round(h3.average_hexagon_edge_length(RES, unit="km"), 3),
        },
        "policies_by_fleet": fleet_rows,
        "batching_window": sorted(window_rows, key=lambda r: r["window_s"]),
        "matching_speed": speed,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "experiments.json").write_text(json.dumps(result, indent=1) + "\n")
    plots(fleet_rows, window_rows, speed)
    for r in sorted(out, key=lambda r: (r["fleet"], r["window_s"])):
        print(
            {
                k: r[k]
                for k in (
                    "policy",
                    "fleet",
                    "completed_pct",
                    "wait_p50_s",
                    "wait_p90_s",
                    "deadhead_share_pct",
                    "utilization_pct",
                    "sim_seconds",
                )
            }
        )


if __name__ == "__main__":
    main()
