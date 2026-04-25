"""geocode london poi names via nominatim, write data/london_pois.json.
runs once at dataset preparation time. requires network access to
nominatim.openstreetmap.org (1 req/s policy enforced).

usage:
    python scripts/seed_pois.py
    python scripts/seed_pois.py --out data/london_pois.json   # default path

skips names that return no result or fall outside the london bounding box.
logs skipped names to stderr, successful geocodes to stdout.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT    = "london-dynamic-routing/0.1 (openreward-hackathon)"
LONDON_BBOX   = {"lat_min": 51.28, "lat_max": 51.70, "lon_min": -0.51, "lon_max": 0.33}
CHARING_CROSS = (51.5074, -0.1278)

# location names to geocode, grouped by category.
# nominatim is queried once per name; results outside LONDON_BBOX are dropped.
CATEGORIES: dict[str, list[str]] = {
    "tube_station": [
        "King's Cross St Pancras station London",
        "Westminster station London",
        "Bank station London",
        "Oxford Circus station London",
        "Liverpool Street station London",
        "Victoria station London",
        "Waterloo station London",
        "London Bridge station London",
        "Paddington station London",
        "Euston station London",
        "Canary Wharf station London",
        "Bond Street station London",
        "Knightsbridge station London",
        "Notting Hill Gate station London",
        "Camden Town station London",
        "Angel station London",
        "Old Street station London",
        "Brixton station London",
        "Stockwell station London",
        "Clapham Common station London",
        "Hammersmith station London",
        "Earl's Court station London",
        "Shepherd's Bush station London",
        "Wembley Park station London",
        "Stratford station London",
        "Greenwich station London",
        "Putney Bridge station London",
        "Walthamstow Central station London",
        "Elephant and Castle station London",
        "Bethnal Green station London",
        "Mile End station London",
        "Canning Town station London",
        "Tottenham Court Road station London",
        "Holborn station London",
        "Farringdon station London",
        "Moorgate station London",
        "Aldgate station London",
        "Tower Hill station London",
        "Blackfriars station London",
        "Embankment station London",
        "Leicester Square station London",
        "Piccadilly Circus station London",
        "Green Park station London",
        "Sloane Square station London",
        "South Kensington station London",
        "Dalston Junction station London",
        "Highbury and Islington station London",
        "Finsbury Park station London",
        "Seven Sisters station London",
        "Shepherd's Bush Market station London",
        "Kensal Rise station London",
        "Kilburn High Road station London",
        "West Hampstead station London",
        "Swiss Cottage station London",
        "St John's Wood station London",
        "Edgware Road station London",
        "Baker Street station London",
        "Great Portland Street station London",
        "Warren Street station London",
        "Goodge Street station London",
        "Covent Garden station London",
        "Vauxhall station London",
        "Kennington station London",
        "Oval station London",
        "Tooting Broadway station London",
        "Tooting Bec station London",
        "Balham station London",
        "Clapham North station London",
        "Clapham South station London",
        "Peckham Rye station London",
        "New Cross station London",
        "Deptford station London",
        "Lewisham station London",
        "Forest Hill station London",
        "Sydenham station London",
        "West Hampstead station London",
        "Kilburn station London",
        "Baker Street station London",
        "Marylebone station London",
    ],
    "hospital": [
        "St Thomas' Hospital London",
        "Guy's Hospital London",
        "King's College Hospital Denmark Hill London",
        "St Mary's Hospital London",
        "Royal Free Hospital London",
        "University College Hospital London",
        "Royal London Hospital London",
        "Lewisham Hospital London",
        "Whittington Hospital London",
        "Homerton Hospital London",
        "Chelsea and Westminster Hospital London",
        "Charing Cross Hospital London",
        "Barts Hospital London",
        "Great Ormond Street Hospital London",
        "Moorfields Eye Hospital London",
        "St George's Hospital London",
        "Queen Elizabeth Hospital Woolwich London",
        "Princess Royal University Hospital London",
    ],
    "shopping": [
        "Westfield London Shepherd's Bush",
        "Westfield Stratford City London",
        "Selfridges London",
        "Harrods London",
        "Borough Market London",
        "Camden Market London",
        "Brick Lane Market London",
        "Covent Garden London",
        "Brixton Market London",
        "Oxford Street London",
        "Carnaby Street London",
        "Spitalfields Market London",
        "Portobello Road Market London",
        "Greenwich Market London",
        "One New Change London",
        "Leadenhall Market London",
        "Petticoat Lane Market London",
        "Ridley Road Market Hackney London",
        "Roman Road Market Bow London",
    ],
    "landmark": [
        "Tower of London",
        "British Museum London",
        "Natural History Museum London",
        "Tate Modern London",
        "London Eye London",
        "Big Ben London",
        "Buckingham Palace London",
        "Trafalgar Square London",
        "Hyde Park London",
        "Regent's Park London",
        "St Paul's Cathedral London",
        "Greenwich Park London",
        "Victoria and Albert Museum London",
        "Science Museum London",
        "National Gallery London",
        "Barbican Centre London",
        "Southbank Centre London",
        "Queen Elizabeth Olympic Park London",
        "Battersea Power Station London",
        "The O2 Arena London",
        "Tower Bridge London",
        "Canary Wharf London",
        "Wembley Stadium London",
        "Emirates Stadium London",
        "Stamford Bridge Chelsea London",
    ],
    "depot": [
        "Stockwell Bus Garage London",
        "Holloway Bus Garage London",
        "West Ham Bus Garage London",
        "Norwood Bus Garage London",
        "Camberwell Bus Garage London",
        "Brixton Bus Garage London",
        "Putney Bus Garage London",
        "Tottenham Bus Garage London",
    ],
    "school": [
        "University College London",
        "King's College London Strand",
        "Imperial College London",
        "London School of Economics London",
        "Queen Mary University of London",
        "City University of London",
        "Goldsmiths University of London",
        "SOAS University of London",
        "Birkbeck University of London",
        "University of Westminster London",
        "London Metropolitan University",
        "London South Bank University",
        "Brunel University London",
        "University of East London",
        "University of Greenwich London",
        "Middlesex University London",
        "University of Roehampton London",
        "Kingston University London",
    ],
    "residential": [
        "Islington London",
        "Hackney London",
        "Camden London",
        "Lambeth London",
        "Southwark London",
        "Tower Hamlets London",
        "Wandsworth London",
        "Lewisham London",
        "Greenwich London",
        "Hammersmith and Fulham London",
        "Kensington London",
        "Chelsea London",
        "Westminster London",
        "Fulham London",
        "Notting Hill London",
        "Shoreditch London",
        "Bermondsey London",
        "Rotherhithe London",
        "Poplar London",
        "Deptford London",
        "New Cross London",
        "Camberwell London",
        "Dulwich London",
        "Streatham London",
        "Tooting London",
        "Balham London",
        "Clapham London",
        "Battersea London",
        "Vauxhall London",
        "Kennington London",
        "Walworth London",
        "Whitechapel London",
        "Wapping London",
        "Stepney Green London",
        "Bow London",
        "Forest Gate London",
        "Plaistow London",
        "Upton Park London",
        "East Ham London",
        "Stoke Newington London",
        "Clapton London",
        "Herne Hill London",
        "Brixton Hill London",
    ],
    "business": [
        "City of London",
        "Mayfair London",
        "Soho London",
        "Clerkenwell London",
        "Paddington Basin London",
        "Nine Elms London",
        "White City London",
        "Stratford International station London",
        "Fleet Street London",
        "Aldgate East London",
        "Euston Road London",
        "Victoria Embankment London",
    ],
}


def zone_for(lat: float, lon: float) -> int:
    """zone 1-4 by straight-line km from charing cross."""
    cx_lat, cx_lon = CHARING_CROSS
    dlat = math.radians(lat - cx_lat)
    dlon = math.radians(lon - cx_lon)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(cx_lat)) * math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2)
    d_km = 2 * 6371.0 * math.asin(math.sqrt(a))
    if d_km < 3:
        return 1
    if d_km < 6:
        return 2
    if d_km < 10:
        return 3
    return 4


def in_bbox(lat: float, lon: float) -> bool:
    b = LONDON_BBOX
    return b["lat_min"] < lat < b["lat_max"] and b["lon_min"] < lon < b["lon_max"]


def geocode(name: str) -> tuple[float, float] | None:
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"q": name, "format": "json", "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        print(f"  geocode error for '{name}': {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/london_pois.json",
                    help="output path (default: data/london_pois.json)")
    args = ap.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_names = sum(len(v) for v in CATEGORIES.values())
    print(f"geocoding {total_names} names via nominatim → {output}")

    pois: list[dict] = []
    done = 0

    for category, names in CATEGORIES.items():
        # deduplicate within category before querying
        seen_in_cat: set[str] = set()
        for name in names:
            if name in seen_in_cat:
                continue
            seen_in_cat.add(name)
            done += 1

            coords = geocode(name)
            time.sleep(1.1)  # nominatim 1 req/s policy

            if coords is None:
                print(f"  [{done}/{total_names}] skip (no result): {name}", file=sys.stderr)
                continue

            lat, lon = coords
            if not in_bbox(lat, lon):
                print(f"  [{done}/{total_names}] skip (outside bbox): {name} "
                      f"lat={lat:.4f} lon={lon:.4f}", file=sys.stderr)
                continue

            pois.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "zone": zone_for(lat, lon),
                "category": category,
            })
            print(f"  [{done}/{total_names}] ok: {name} zone={pois[-1]['zone']}")

    # deduplicate across categories by name (keep first occurrence)
    seen_names: set[str] = set()
    unique_pois = []
    for p in pois:
        if p["name"] not in seen_names:
            seen_names.add(p["name"])
            unique_pois.append(p)

    removed = len(pois) - len(unique_pois)
    if removed:
        print(f"  removed {removed} cross-category duplicates")

    output.write_text(json.dumps(unique_pois, indent=2))
    print(f"\nwrote {len(unique_pois)} pois → {output}")


if __name__ == "__main__":
    main()
