"""Dispatch simulation: drivers, riders and a matching policy, stepped one second at a time.

A rider requests at a recorded time and waits up to PATIENCE seconds to be matched; once matched, the driver drives
to the pickup (deadhead), carries the rider for the trip's recorded duration, and becomes idle at the drop-off.
Idle drivers are indexed by H3 cell, so finding nearby drivers looks at a few hexagons, not the whole fleet.

Policies
  greedy        match each request the moment it arrives to the idle driver with the shortest pickup ETA
  batched(W)    every W seconds, match all waiting riders to idle drivers at once with a minimum-total-ETA
                assignment (Hungarian algorithm); unmatched riders keep waiting until their patience runs out
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import h3
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .data import haversine_km

RES = 8  # H3 resolution: hexagons with ~0.46 km edges
PATIENCE = 180.0  # seconds a rider waits to be matched before giving up
MAX_ETA = 600.0  # never send a driver more than 10 minutes away
NO_EDGE = 1e9


@dataclass
class Travel:
    speed_kmh: float
    detour: float

    def eta(self, lat1, lon1, lat2, lon2):
        return haversine_km(lat1, lon1, lat2, lon2) * self.detour / self.speed_kmh * 3600

    def ring_limit(self, max_eta: float = MAX_ETA) -> int:
        """H3 rings needed to cover every driver within max_eta (with a margin for hexagon shape)."""
        reach_km = max_eta / 3600 * self.speed_kmh / self.detour
        edge_km = h3.average_hexagon_edge_length(RES, unit="km")
        return int(np.ceil(reach_km / (1.5 * edge_km))) + 1


class DriverIndex:
    """Idle drivers by H3 cell."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat, self.lon = lat, lon
        self.cell = [h3.latlng_to_cell(a, b, RES) for a, b in zip(lat, lon, strict=True)]
        self.idle: dict[str, set[int]] = defaultdict(set)

    def add(self, d: int, lat: float, lon: float) -> None:
        self.lat[d], self.lon[d] = lat, lon
        self.cell[d] = h3.latlng_to_cell(lat, lon, RES)
        self.idle[self.cell[d]].add(d)

    def remove(self, d: int) -> None:
        self.idle[self.cell[d]].discard(d)

    def near(self, lat: float, lon: float, k: int) -> list[int]:
        out: list[int] = []
        for c in h3.grid_disk(h3.latlng_to_cell(lat, lon, RES), k):
            out.extend(self.idle.get(c, ()))
        return out


@dataclass
class Result:
    policy: str
    fleet: int
    requests: int
    completed: int
    unfulfilled: int
    match_wait_s: list[float] = field(default_factory=list)  # request -> match
    pickup_eta_s: list[float] = field(default_factory=list)  # match -> pickup
    deadhead_km: float = 0.0
    trip_km: float = 0.0
    busy_s: float = 0.0
    duration_s: float = 0.0
    match_ms: list[float] = field(default_factory=list)  # compute time per matching call

    def summary(self) -> dict:
        total = np.array(self.match_wait_s) + np.array(self.pickup_eta_s)
        return {
            "policy": self.policy,
            "fleet": self.fleet,
            "requests": self.requests,
            "completed_pct": round(100 * self.completed / self.requests, 1),
            "unfulfilled_pct": round(100 * self.unfulfilled / self.requests, 1),
            "wait_p50_s": round(float(np.median(total)), 1) if len(total) else None,
            "wait_p90_s": round(float(np.percentile(total, 90)), 1) if len(total) else None,
            "pickup_eta_mean_s": round(float(np.mean(self.pickup_eta_s)), 1) if self.pickup_eta_s else None,
            "match_wait_mean_s": round(float(np.mean(self.match_wait_s)), 1) if self.match_wait_s else None,
            "deadhead_share_pct": round(float(100 * self.deadhead_km / max(self.deadhead_km + self.trip_km, 1e-9)), 1),
            "utilization_pct": round(float(100 * self.busy_s / (self.fleet * self.duration_s)), 1),
            "match_ms_p50": round(float(np.median(self.match_ms)), 3) if self.match_ms else None,
            "match_ms_p99": round(float(np.percentile(self.match_ms, 99)), 3) if self.match_ms else None,
        }


def simulate(
    req: pd.DataFrame,
    travel: Travel,
    fleet: int,
    window: float = 0.0,
    seed: int = 0,
    brute_force: bool = False,
    start_positions: np.ndarray | None = None,
    max_eta: float = MAX_ETA,
) -> Result:
    """window = 0 is greedy; window > 0 is batched matching every `window` seconds.
    Drivers start at `start_positions` (fleet x [lat, lon]) or, by default, at random requested pickup points, so
    supply starts out shaped like demand."""
    rng = np.random.default_rng(seed)
    if start_positions is None:
        start = rng.choice(len(req), fleet, replace=len(req) < fleet)
        start_positions = np.column_stack([req.plat.to_numpy()[start], req.plon.to_numpy()[start]])
    index = DriverIndex(start_positions[:, 0].astype(float).copy(), start_positions[:, 1].astype(float).copy())
    for d in range(fleet):
        index.add(d, index.lat[d], index.lon[d])
    free_at: list[tuple[float, int, float, float]] = []  # (time, driver, lat, lon) for drivers on a job
    k_max = travel.ring_limit(max_eta)
    name = "greedy" if window == 0 else f"batched {window:g}s"
    if max_eta != MAX_ETA:
        name += f", pickup cap {max_eta / 60:g} min"
    res = Result(name, fleet, len(req), 0, 0)
    t_req, plat, plon = req.t.to_numpy(), req.plat.to_numpy(), req.plon.to_numpy()
    dlat, dlon, trip_s, trip_km = (
        req.dlat.to_numpy(),
        req.dlon.to_numpy(),
        req.trip_s.to_numpy(),
        req.trip_km.to_numpy(),
    )
    horizon = float(t_req.max()) + PATIENCE + 1
    waiting: list[int] = []
    nxt = 0
    rider_cell = [h3.latlng_to_cell(a, b, RES) for a, b in zip(plat, plon, strict=True)]

    def dispatch(r: int, d: int, now: float, eta: float) -> None:
        index.remove(d)
        res.completed += 1
        res.match_wait_s.append(now - t_req[r])
        res.pickup_eta_s.append(eta)
        res.deadhead_km += eta / 3600 * travel.speed_kmh
        res.trip_km += trip_km[r]
        done = now + eta + trip_s[r]
        res.busy_s += min(done, horizon) - now
        free_at.append((done, d, dlat[r], dlon[r]))

    def candidates(r: int) -> np.ndarray:
        if brute_force:
            return np.array(sorted(d for s in index.idle.values() for d in s), dtype=int)
        return np.array(index.near(plat[r], plon[r], k_max), dtype=int)

    now = 0.0
    step = window if window > 0 else 1.0
    while now <= horizon:
        freed_cells: set[str] = set()
        if free_at:  # drivers finishing a trip become idle where they dropped off
            still = []
            for item in free_at:
                if item[0] <= now:
                    index.add(item[1], item[2], item[3])
                    freed_cells.add(index.cell[item[1]])
                else:
                    still.append(item)
            free_at[:] = still
        old = [r for r in waiting if now - t_req[r] <= PATIENCE]  # riders who give up leave the queue
        res.unfulfilled += len(waiting) - len(old)
        new = []
        while nxt < len(t_req) and t_req[nxt] <= now:
            new.append(nxt)
            nxt += 1
        waiting = old + new
        arrived = len(old)  # waiting[:arrived] already failed to match; waiting[arrived:] are new this step
        if waiting:
            t0 = time.perf_counter()
            if window == 0:  # greedy, in arrival order
                # a rider who already failed can only match now if a driver was freed within its search radius
                reach = set().union(*(h3.grid_disk(c, k_max) for c in freed_cells)) if freed_cells else set()
                left = []
                for pos_in_queue, r in enumerate(waiting):  # keep first-come, first-served order
                    if pos_in_queue < arrived and rider_cell[r] not in reach:
                        left.append(r)
                        continue
                    cand = candidates(r)
                    if len(cand):
                        etas = travel.eta(plat[r], plon[r], index.lat[cand], index.lon[cand])
                        j = int(np.argmin(etas))
                        if etas[j] <= max_eta:
                            dispatch(r, int(cand[j]), now, float(etas[j]))
                            continue
                    left.append(r)
                waiting = left
            else:  # batched: one assignment over every waiting rider and every nearby idle driver
                cand_sets = [candidates(r) for r in waiting]
                drivers = np.unique(np.concatenate(cand_sets)) if any(len(c) for c in cand_sets) else np.array([], int)
                if len(drivers):
                    cost = np.full((len(waiting), len(drivers)), NO_EDGE)
                    pos = {d: i for i, d in enumerate(drivers)}
                    for i, (r, cand) in enumerate(zip(waiting, cand_sets, strict=True)):
                        if len(cand):
                            etas = travel.eta(plat[r], plon[r], index.lat[cand], index.lon[cand])
                            ok = etas <= max_eta
                            cost[i, [pos[d] for d in cand[ok]]] = etas[ok]
                    rows, cols = linear_sum_assignment(cost)
                    matched = set()
                    for i, j in zip(rows, cols, strict=True):
                        if cost[i, j] < NO_EDGE:
                            dispatch(waiting[i], int(drivers[j]), now, float(cost[i, j]))
                            matched.add(i)
                    waiting = [r for i, r in enumerate(waiting) if i not in matched]
            res.match_ms.append((time.perf_counter() - t0) * 1000)
        if nxt >= len(t_req) and not waiting:
            break
        now += step
    res.unfulfilled += len(waiting)
    res.duration_s = horizon
    return res
