import h3
import numpy as np
import pandas as pd
import pytest

from dispatchsim.data import haversine_km
from dispatchsim.sim import MAX_ETA, PATIENCE, RES, DriverIndex, Travel, simulate

TRAVEL = Travel(speed_kmh=15.0, detour=1.2)
MIDTOWN = (40.754, -73.984)


def requests(rows) -> pd.DataFrame:
    """rows: (t, plat, plon[, trip_s]); drop-off back at the pickup point."""
    out = [
        {
            "t": r[0],
            "plat": r[1],
            "plon": r[2],
            "dlat": r[1],
            "dlon": r[2],
            "trip_s": r[3] if len(r) > 3 else 600.0,
            "trip_km": 2.0,
        }
        for r in rows
    ]
    return pd.DataFrame(out)


def test_haversine_known_distance():
    # Times Square to the Empire State Building is about 1.1 km in a straight line
    assert haversine_km(40.7580, -73.9855, 40.7484, -73.9857) == pytest.approx(1.07, abs=0.05)


def test_ring_limit_reaches_every_driver_within_the_eta_cap():
    rng = np.random.default_rng(0)
    k = TRAVEL.ring_limit()
    origin = h3.latlng_to_cell(*MIDTOWN, RES)
    disk = set(h3.grid_disk(origin, k))
    for _ in range(2000):
        lat = MIDTOWN[0] + rng.uniform(-0.03, 0.03)
        lon = MIDTOWN[1] + rng.uniform(-0.04, 0.04)
        if TRAVEL.eta(*MIDTOWN, lat, lon) <= MAX_ETA:
            assert h3.latlng_to_cell(lat, lon, RES) in disk


def test_driver_index_tracks_idle_drivers_by_cell():
    ix = DriverIndex(np.array([MIDTOWN[0]]), np.array([MIDTOWN[1]]))
    ix.add(0, *MIDTOWN)
    assert ix.near(*MIDTOWN, 1) == [0]
    ix.remove(0)
    assert ix.near(*MIDTOWN, 1) == []


def test_batched_matching_minimizes_total_eta_where_greedy_does_not():
    # Two riders and two drivers on one street. Rider A is first in line, between the drivers; rider B is just past
    # driver 1. Greedy gives A driver 1 (its nearest) and leaves B with distant driver 2; the batch assignment sends
    # driver 2 to A and driver 1 to B, for a shorter total pickup time.
    lat = MIDTOWN[0]
    d1, d2 = MIDTOWN[1], MIDTOWN[1] + 0.012
    reqs = requests([(0.0, lat, d1 + 0.004), (0.0, lat, d1 - 0.004)])
    drivers = pd.DataFrame({"plat": [lat, lat], "plon": [d1, d2]})

    def run(window):
        return simulate(reqs, TRAVEL, 2, window, seed=0, start_positions=drivers[["plat", "plon"]].to_numpy())

    greedy, batched = run(0), run(10)
    assert greedy.completed == batched.completed == 2
    assert sum(batched.pickup_eta_s) < sum(greedy.pickup_eta_s)


def test_riders_give_up_after_their_patience_when_no_driver_is_near():
    far = (40.80, -73.95)  # about 6 km north: beyond the 10-minute cap
    reqs = requests([(0.0, *MIDTOWN)])
    r = simulate(reqs, TRAVEL, 1, 0, start_positions=np.array([far]))
    assert (r.completed, r.unfulfilled) == (0, 1)
    assert r.duration_s >= PATIENCE


def test_h3_candidate_search_matches_brute_force():
    rng = np.random.default_rng(3)
    rows = [
        (
            float(t),
            MIDTOWN[0] + rng.uniform(-0.02, 0.02),
            MIDTOWN[1] + rng.uniform(-0.02, 0.02),
            float(rng.uniform(300, 900)),
        )
        for t in np.sort(rng.uniform(0, 1200, 300))
    ]
    reqs = requests(rows)
    for window in (0, 15):
        a = simulate(reqs, TRAVEL, 60, window, seed=1).summary()
        b = simulate(reqs, TRAVEL, 60, window, seed=1, brute_force=True).summary()
        assert {k: v for k, v in a.items() if not k.startswith("match_ms")} == {
            k: v for k, v in b.items() if not k.startswith("match_ms")
        }
