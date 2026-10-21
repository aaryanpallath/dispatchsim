# dispatchsim — Ride Dispatch Simulator on Real NYC Demand
Tech: Python, NumPy, SciPy, H3, pandas, Matplotlib, pytest, GitHub Actions
- Built a ride-hailing dispatch simulator on 61,796 real NYC taxi requests, with an H3 spatial index of idle drivers and greedy vs batched (Hungarian assignment) matching.
- Showed batched matching completed 83.5% of rides vs 68.8% for greedy at 4,000 drivers, cutting median wait from 7.7 to 1.6 minutes and empty driving from 29% to 11% of distance.
- Traced greedy's losses to long-distance pickups and confirmed it by experiment; benchmarked matching to 50,000 drivers and found vectorized brute force beat my H3 search, reporting it as a negative result.
