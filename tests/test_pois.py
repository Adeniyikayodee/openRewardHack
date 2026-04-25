"""tests for seed_pois — network calls are monkey-patched via pytest monkeypatch."""
import json
from pathlib import Path
from unittest.mock import MagicMock

from scripts.seed_pois import geocode, in_bbox, zone_for

POI_FILE = Path("data/london_pois.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nominatim_stub(lat: float, lon: float) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = [{"lat": str(lat), "lon": str(lon)}]
    return mock


def _load():
    return json.loads(POI_FILE.read_text())


# ---------------------------------------------------------------------------
# geocode() — network stubbed
# ---------------------------------------------------------------------------

def test_geocode_returns_coords(monkeypatch):
    monkeypatch.setattr(
        "scripts.seed_pois.requests.get",
        lambda *a, **kw: _nominatim_stub(51.5074, -0.1278),
    )
    assert geocode("King's Cross London") == (51.5074, -0.1278)


def test_geocode_returns_none_on_empty_result(monkeypatch):
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = []
    monkeypatch.setattr("scripts.seed_pois.requests.get", lambda *a, **kw: mock)
    assert geocode("nonexistent place xyz") is None


def test_geocode_returns_none_on_network_error(monkeypatch):
    def raise_error(*a, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("scripts.seed_pois.requests.get", raise_error)
    assert geocode("anything") is None


# ---------------------------------------------------------------------------
# pure computation — no network
# ---------------------------------------------------------------------------

def test_in_bbox_central_london():
    assert in_bbox(51.5074, -0.1278)  # charing cross


def test_in_bbox_rejects_paris():
    assert not in_bbox(48.8566, 2.3522)


def test_zone_for_zone1():
    # charing cross itself → zone 1 (< 3 km)
    assert zone_for(51.5074, -0.1278) == 1


def test_zone_for_zone4():
    # outer east london
    assert zone_for(51.65, 0.25) == 4


# ---------------------------------------------------------------------------
# data-quality tests against generated data/london_pois.json
# ---------------------------------------------------------------------------

def test_poi_file_exists():
    assert POI_FILE.exists(), "data/london_pois.json not found — run: python scripts/seed_pois.py"


def test_poi_count():
    pois = _load()
    assert len(pois) >= 80, f"only {len(pois)} pois, need >= 80"


def test_poi_inside_london_bbox():
    pois = _load()
    for p in pois:
        assert 51.28 < p["lat"] < 51.70, f"lat out of range: {p}"
        assert -0.51 < p["lon"] < 0.33, f"lon out of range: {p}"


def test_poi_required_fields():
    pois = _load()
    for p in pois:
        for field in ("name", "lat", "lon", "zone", "category"):
            assert field in p, f"missing field '{field}' in {p}"


def test_poi_categories_diverse():
    pois = _load()
    present = {p["category"] for p in pois}
    missing = {"tube_station", "hospital", "depot", "landmark"} - present
    assert not missing, f"missing required categories: {missing}"


def test_poi_all_zones_covered():
    pois = _load()
    zones = {p["zone"] for p in pois}
    assert {1, 2, 3} <= zones, f"missing zones: {{1,2,3}} - {zones}"


def test_poi_no_duplicates():
    pois = _load()
    names = [p["name"] for p in pois]
    assert len(names) == len(set(names)), "duplicate poi names found"


def test_poi_zone_values_valid():
    pois = _load()
    for p in pois:
        assert p["zone"] in (1, 2, 3, 4), f"invalid zone {p['zone']} in {p}"
