"""Download OpenStreetMap features for the ML hazard model.  RUN ON YOUR LAPTOP (needs internet).

Uses the public Overpass API (free; be patient - each query can take 30-120 s). Writes compact JSON to data/osm/:
  rivers.json      rivers, streams, canals          (line vertices)
  drains.json      mapped drains and ditches        (line vertices)
  roads.json       drivable roads                   (line vertices)  -> built-up / paved-surface proxy
  informal.json    areas tagged as informal settlements (polygon rings)
Each file records the query and download date, so the data source is citable (OpenStreetMap contributors, ODbL).

usage: python fetch_osm.py
"""
import json, os, time, datetime
import urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "osm")
BBOX = "-1.45,36.60,-1.10,37.10"   # south, west, north, east (Nairobi, matches the hazard maps)
URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
UA = "nairobi-flood-cat/0.1 (student hackathon project)"

QUERIES = {
    "rivers":   f'way["waterway"~"^(river|stream|canal)$"]({BBOX});',
    "drains":   f'way["waterway"~"^(drain|ditch)$"]({BBOX});',
    "roads":    f'way["highway"~"^(primary|secondary|tertiary|residential|unclassified|living_street|service)$"]({BBOX});',
    "informal": f'(way["residential"="informal"]({BBOX});way["informal"="yes"]({BBOX});'
                f'relation["residential"="informal"]({BBOX}););',
}


def overpass(q):
    body = urllib.parse.urlencode({"data": f"[out:json][timeout:180];{q}out geom;"}).encode()
    last = None
    for url in URLS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=body, headers={"User-Agent": UA})
                return json.load(urllib.request.urlopen(req, timeout=240))
            except Exception as e:
                last = e
                print(f"    {url.split('/')[2]} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {last}")


def lines(res, step=1):
    pts = []
    for el in res.get("elements", []):
        for g in el.get("geometry", [])[::step]:
            pts.append([round(g["lat"], 6), round(g["lon"], 6)])
    return pts


def rings(res):
    out = []
    for el in res.get("elements", []):
        if el["type"] == "way" and el.get("geometry"):
            out.append([[g["lat"], g["lon"]] for g in el["geometry"]])
        for m in el.get("members", []):
            if m.get("role") == "outer" and m.get("geometry"):
                out.append([[g["lat"], g["lon"]] for g in m["geometry"]])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, q in QUERIES.items():
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            print(f"have  {name}"); continue
        print(f"fetching {name} ...")
        res = overpass(q)
        data = rings(res) if name == "informal" else lines(res, step=1 if name != "roads" else 2)
        json.dump({"source": "OpenStreetMap contributors (ODbL), via Overpass API", "query": q,
                   "downloaded": datetime.date.today().isoformat(), "n": len(data), "data": data},
                  open(path, "w"))
        print(f"  saved {name}: {len(data)} {'polygons' if name == 'informal' else 'points'}")
        time.sleep(5)


if __name__ == "__main__":
    main()
