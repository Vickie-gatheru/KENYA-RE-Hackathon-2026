"""AI step 2: give each extracted place coordinates.  RUN THIS ON YOUR LAPTOP (needs internet).

Uses OpenStreetMap Nominatim (free; max 1 request/second; results cached in data/gazetteer.csv so each
place is looked up once). Places that fail or land outside Nairobi are left blank for manual entry -
open data/gazetteer.csv, fill lat/lon from a map, set method=manual.

Independence rule: we never take coordinates from nairobi_hotspots_geocoded.csv, even for a place with
the same name, so the validation set stays separate from the model inputs.

usage: python geocode.py            -> reads out/signals_raw.json, writes data/signals.csv
"""
import csv, json, os, time
import urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GAZ = os.path.join(HERE, "data", "gazetteer.csv")
NAIROBI_BBOX = (36.60, -1.45, 37.10, -1.10)  # lon_min, lat_min, lon_max, lat_max
UA = "nairobi-flood-cat-hackathon/0.1 (student project)"


def nominatim(place, city="Nairobi", bbox=NAIROBI_BBOX, country="Kenya"):
    """bbox = (lon_min, lat_min, lon_max, lat_max); results outside it are not returned."""
    q = urllib.parse.urlencode({"q": f"{place}, {city}, {country}", "format": "json", "limit": 1,
                                "viewbox": ",".join(map(str, bbox)), "bounded": 1})
    req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.load(r)
    time.sleep(1.1)
    return (float(res[0]["lat"]), float(res[0]["lon"]), res[0].get("display_name", "")) if res else None


def main():
    sig = json.load(open(os.path.join(HERE, "out", "signals_raw.json")))["signals"]
    os.makedirs(os.path.dirname(GAZ), exist_ok=True)
    gaz = {}
    if os.path.exists(GAZ):
        gaz = {r["place_name"]: r for r in csv.DictReader(open(GAZ, encoding="utf-8"))}
    for name in sorted({s["place_name"] for s in sig} - set(gaz)):
        try:
            hit = nominatim(name)
        except Exception as e:
            print(f"  {name}: error {e}"); hit = None
        gaz[name] = dict(place_name=name, lat=hit[0] if hit else "", lon=hit[1] if hit else "",
                         matched=hit[2] if hit else "", method="nominatim" if hit else "MISSING")
        print(f"  {name}: {gaz[name]['method']} {gaz[name]['matched'][:60]}")
    with open(GAZ, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["place_name", "lat", "lon", "matched", "method"])
        w.writeheader(); w.writerows(gaz.values())
    rows, missing = [], 0
    for s in sig:
        g = gaz[s["place_name"]]
        if not g["lat"]:
            missing += 1; continue
        rows.append({**s, "lat": g["lat"], "lon": g["lon"], "geocode_method": g["method"]})
    out = os.path.join(HERE, "data", "signals.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["place_name"])
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} signals with coordinates -> {out}; {missing} skipped (fill them in {GAZ})")


if __name__ == "__main__":
    main()
