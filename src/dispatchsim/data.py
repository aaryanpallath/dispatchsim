"""Real demand: NYC TLC yellow-taxi trips (March 2016) as ride requests.

The TLC publishes these trips with pickup and drop-off taxi *zones*, not coordinates, so each trip's endpoints are
sampled uniformly inside its zone polygon (Manhattan zones are a few blocks across). Request times and trip
durations are the recorded ones. Travel speed and the road-detour factor are calibrated from the same trips."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2016-03.parquet"
TRIPS_SHA256 = "7e60732898f29c05e0e86a9225207288a5adc270ced66b1f1311cbf5d96f8d83"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
ZONES_SHA256 = "f6d711917bb4340f8f644d5366c51665489eb2d426dd1a4a55677721ae5adf17"
TO_WGS84 = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)  # NY State Plane (feet) -> lat/lon


def fetch(url: str, sha256: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    if hashlib.sha256(dest.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{dest} does not match its published checksum")
    return dest


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


@dataclass
class Zones:
    polygons: dict[int, Polygon]  # in State Plane feet
    borough: dict[int, str]

    def sample(self, zone_ids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """A uniform random point inside each zone, as (lat, lon) arrays."""
        xs, ys = np.empty(len(zone_ids)), np.empty(len(zone_ids))
        for i, z in enumerate(zone_ids):
            poly = self.polygons[int(z)]
            x0, y0, x1, y1 = poly.bounds
            while True:  # rejection sampling inside the bounding box
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                if poly.contains(Point(x, y)):
                    xs[i], ys[i] = x, y
                    break
        lon, lat = TO_WGS84.transform(xs, ys)
        return np.asarray(lat), np.asarray(lon)


def load_zones(data_dir: Path) -> Zones:
    zip_path = fetch(ZONES_URL, ZONES_SHA256, data_dir / "taxi_zones.zip")
    shp = data_dir / "taxi_zones" / "taxi_zones" / "taxi_zones.shp"
    if not shp.exists():
        import zipfile

        zipfile.ZipFile(zip_path).extractall(data_dir / "taxi_zones")
    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    polygons, borough = {}, {}
    for rec, shape in zip(reader.records(), reader.shapes(), strict=True):
        attrs = dict(zip(fields, rec, strict=True))
        parts = list(shape.parts) + [len(shape.points)]
        rings = [Polygon(shape.points[parts[i] : parts[i + 1]]) for i in range(len(parts) - 1)]
        zid = int(attrs["LocationID"])
        polygons[zid] = unary_union([r.buffer(0) for r in rings])
        borough[zid] = attrs["borough"]
    return Zones(polygons, borough)


@dataclass
class Demand:
    requests: pd.DataFrame  # t (s from start), plat, plon, dlat, dlon, trip_s, trip_km
    speed_kmh: float
    detour: float


def load_demand(data_dir: Path, start: str, hours: float, borough: str = "Manhattan", seed: int = 0) -> Demand:
    trips_path = fetch(TRIPS_URL, TRIPS_SHA256, data_dir / "yellow_tripdata_2016-03.parquet")
    zones = load_zones(data_dir)
    in_borough = {z for z, b in zones.borough.items() if b == borough}
    t0 = pd.Timestamp(start)
    cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance"]
    df = pq.read_table(
        trips_path,
        columns=cols,
        filters=[("tpep_pickup_datetime", ">=", t0), ("tpep_pickup_datetime", "<", t0 + pd.Timedelta(hours=hours))],
    ).to_pandas()
    df = df[df.PULocationID.isin(in_borough) & df.DOLocationID.isin(in_borough)]
    dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds()
    df = df[(dur >= 60) & (dur <= 7200) & (df.trip_distance > 0.05) & (df.trip_distance < 30)].copy()
    df["trip_s"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds()
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    plat, plon = zones.sample(df.PULocationID.to_numpy(), rng)
    dlat, dlon = zones.sample(df.DOLocationID.to_numpy(), rng)
    req = pd.DataFrame(
        {
            "t": (df.tpep_pickup_datetime - t0).dt.total_seconds().to_numpy(),
            "plat": plat,
            "plon": plon,
            "dlat": dlat,
            "dlon": dlon,
            "trip_s": df.trip_s.to_numpy(),
            "trip_km": df.trip_distance.to_numpy() * 1.609344,
        }
    )
    straight = haversine_km(req.plat, req.plon, req.dlat, req.dlon)
    longish = straight > 1.0  # short trips make the detour ratio noisy
    detour = float(np.median(req.trip_km[longish] / straight[longish]))
    speed = float(np.median(req.trip_km / (req.trip_s / 3600)))
    return Demand(req, speed, detour)
