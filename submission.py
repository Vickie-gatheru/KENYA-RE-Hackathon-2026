"""Read a broker's placement submission (PDF or text) and check it before pricing.

  read_text(data)          PDF bytes -> text, page by page (pypdf).
  extract(text, call)      finds the line that states each fact. Rules find labelled lines ("GPS COORDINATES: ...");
                           if an LLM is configured it finds lines for facts the rules missed, plus statements that
                           contradict each other. Either way the QUOTE must appear word for word in the document and
                           the VALUE is parsed from the quote by code - the LLM never supplies a number.
  checks(f, text)          rule-based consistency checks on the broker's own figures, and against map data
                           (OpenStreetMap places and waterways).
  building_events(...)     ASSUMED adjustment for tall buildings and basements: water at street level damages the
                           ground floor and floods the basements; upper floors stay dry.
  apply(o, f, ...)         puts the submission into an evaluation from evaluate.evaluate: building-adjusted losses,
                           checks, and a broker-says / model-says comparison.
"""
import io
import json
import math
import os
import re
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

import catmodel as cm

HERE = os.path.dirname(os.path.abspath(__file__))
BBOX = (-1.45, -1.10, 36.60, 37.10)
BASEMENT_TRIGGER_M = 0.05   # ASSUMED: any water at street level reaches the basements (ramps, stairs, vents)
BASEMENT_FILL_M = 3.0       # ASSUMED: a flooded basement fills about one storey deep


# ---------------------------------------------------------------- 1. text
def read_text(data):
    """PDF bytes (or a path) -> (full text, list of page texts). Plain text is returned as one page."""
    if isinstance(data, str) and os.path.exists(data):
        data = open(data, "rb").read()
    if isinstance(data, bytes) and data[:4] == b"%PDF":
        import pypdf
        pages = [p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(data)).pages]
    else:
        pages = [data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)]
    return "\n".join(pages), pages


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _page(quote, pages):
    q = _norm(quote)
    return next((i + 1 for i, p in enumerate(pages) if q in _norm(p)), None)


# ---------------------------------------------------------------- 2. parsers (value from a quote)
_NUM = r"\d[\d,]*(?:\.\d+)?"


def _nums(s):
    return [float(x.replace(",", "")) for x in re.findall(_NUM, s)]


def _first(s):
    n = _nums(s)
    return n[0] if n else None


def _money(s):
    m = re.search(r"(?:KES|KSh|Kshs?)\.?\s*(" + _NUM + ")", s, re.I)
    return float(m.group(1).replace(",", "")) if m else _first(s)


def _after_colon(s):
    return s.split(":", 1)[1].strip() if ":" in s else s.strip()


def _coords(s):
    m = re.search(r"(-?\d{1,2}\.\d+)\s*°?\s*([NS])?\s*,?\s*(-?\d{2,3}\.\d+)\s*°?\s*([EW])?", s)
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(3))
    lat = -abs(lat) if m.group(2) == "S" else lat
    lon = -abs(lon) if m.group(4) == "W" else lon
    return lat, lon


def _floors(s):
    above = re.search(r"(\d+)\s*\(?\s*above", s, re.I)
    base = re.search(r"(\d+)\s*\(?\s*basement", s, re.I)
    if not above:
        n = _first(s)
        return (int(n), int(base.group(1)) if base else 0) if n else None
    return int(above.group(1)), int(base.group(1)) if base else 0


def _date(s):
    m = re.search(r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})", s)
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(m.group(1), fmt).date() if m else None
        except ValueError:
            pass
    return None


def _deductible(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    k = re.search(r"(?:KES|KSh)\s*(" + _NUM + ")", s, re.I)
    return (float(m.group(1)), float(k.group(1).replace(",", "")) if k else 0.0) if m else None


def _river(s):
    km = re.search(r"(" + _NUM + r")\s*km", s)
    dirn = re.search(r"\(([A-Za-z\- ]+)\)", s)
    name = re.search(r"([A-Z][A-Za-z]+ River)", s)
    return dict(km=float(km.group(1)), direction=dirn.group(1) if dirn else None,
                name=name.group(1) if name else "river") if km else None


def _construction(s):
    t = s.lower()
    if re.search(r"\b(rcc|reinforced concrete|concrete frame|steel frame)\b", t):
        return "concrete_rcc"
    if re.search(r"\b(iron sheet|mabati|informal)\b", t):
        return "informal_iron_sheet"
    if re.search(r"\b(mud|timber|semi-permanent|wattle)\b", t):
        return "semi_permanent"
    if re.search(r"\b(masonry|brick|stone|block)\b", t):
        return "permanent_masonry"
    return None


# field: (what it is - for the LLM, rule regex that finds the line, parser)
FIELDS = {
    "client": ("the insured / property name", r"^(?:CLIENT|INSURED|PROPERTY NAME)\s*:", _after_colon),
    "address": ("the street address or area", r"^(?:STREET )?ADDRESS\s*:", _after_colon),
    "coords": ("the building's GPS coordinates", r"^(?:GPS|COORDINATES|LOCATION COORDINATES)", _coords),
    "construction": ("the construction / structural classification", r"^CONSTRUCTION[A-Z ]*:", _after_colon),
    "floors": ("number of floors above ground and basements", r"number of (?:floors|storeys)", _floors),
    "gfa_m2": ("the gross floor area", r"^GROSS FLOOR AREA", _first),
    "height_m": ("the building height to the roof", r"building height \(to roof", _first),
    "storey_m": ("the typical floor-to-floor height", r"floor-to-floor height", _first),
    "ground_storey_m": ("the ground floor storey height", r"ground to first floor", _first),
    "elevation_m": ("the site elevation", r"^ELEVATION\s*:", _first),
    "river": ("distance and direction to the nearest river", r"proximity to .*river", _river),
    "river_elevation_m": ("the river's elevation", r"river elevation", _first),
    "tiv_kes": ("the total insured value / sum insured in KES", r"(?:\bTIV\b|sum insured|total insured value).*KES",
                _money),
    "water_daily_l": ("daily water consumption", r"daily consumption", _first),
    "water_monthly_m3": ("monthly water consumption", r"monthly water", _first),
    "fire_tank_l": ("the fire water tank size", r"fire water (?:tank|supply)", _first),
    "sprinkler_m3h": ("the sprinkler system flow rate", r"^-?\s*system capacity.*m³/h", _first),
    "sump_m3h": ("the basement sump pump capacity", r"sump pump capacity", _first),
    "issued": ("the date the submission was issued", r"^DATE ISSUED", _date),
    "expiry": ("the date the offer expires", r"^EXPIRY", _date),
    "flood_deductible": ("the proposed flood deductible", r"%\s*deductible", _deductible),
}


# ---------------------------------------------------------------- 3. rule extraction
def _lines(text):
    return [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]


def _header(lines, i):
    """The nearest section header above line i (an upper-case line ending in ':')."""
    for j in range(i - 1, max(-1, i - 12), -1):
        l = lines[j]
        if l.endswith(":") and re.sub(r"[^A-Za-z]", "", l).isupper():
            return l
    return ""


def _areas(lines, n_basements):
    """The floor-area breakdown under GROSS FLOOR AREA: label, m2, how many floors it covers, kind."""
    out = []
    start = next((i for i, l in enumerate(lines) if re.match(r"^GROSS FLOOR AREA", l, re.I)), None)
    if start is None:
        return out
    for l in lines[start + 1:start + 12]:
        m = re.match(r"^-\s*(.+?):\s*(" + _NUM + r")\s*m(?:²|2)?(\s*each)?", l)
        if not m:
            if out and l and not l.startswith("-"):
                break
            continue
        label, m2, each = m.group(1), float(m.group(2).replace(",", "")), bool(m.group(3))
        rng = re.search(r"F(\d+)\s*-\s*F(\d+)", label)
        kind = "basement" if "basement" in label.lower() else "ground" if "ground" in label.lower() else "upper"
        n = int(rng.group(2)) - int(rng.group(1)) + 1 if rng else (n_basements if kind == "basement" and each else 1)
        out.append(dict(label=label, m2=m2, n=max(n, 1), kind=kind, quote=l))
    return out


_DIRS = {"north": 0, "northnortheast": 22.5, "northeast": 45, "eastnortheast": 67.5, "east": 90,
         "eastsoutheast": 112.5, "southeast": 135, "southsoutheast": 157.5, "south": 180, "southsouthwest": 202.5,
         "southwest": 225, "westsouthwest": 247.5, "west": 270, "westnorthwest": 292.5, "northwest": 315,
         "northnorthwest": 337.5}


def _bearing_of(word):
    return _DIRS.get(re.sub(r"[^a-z]", "", (word or "").lower()))


def _landmarks(lines):
    out, on = [], False
    for l in lines:
        if re.match(r"^NEARBY LANDMARKS", l, re.I):
            on = True
            continue
        if on:
            m = re.match(r"^-\s*(.+?)\s*\(([A-Za-z\- ]+)\)\s*-\s*(" + _NUM + r")\s*km", l)
            if m:
                out.append(dict(name=m.group(1), direction=m.group(2), km=float(m.group(3)), quote=l))
            elif out and l and not l.startswith("-"):
                break
    return out


_PLANT = [(r"generator", "Generator"), (r"transformer|substation", "Transformer"), (r"chiller", "Chiller plant"),
          (r"transfer switch|\bATS\d?\b", "Transfer switch"), (r"(?<!sump )pumps?\b|pumping", "Water pumps"),
          (r"treatment (?:unit|plant)", "Wastewater plant"), (r"fuel tank", "Fuel tank"),
          (r"\bUPS\b|server|data cent", "IT / UPS")]
_BASEMENT = r"\b(B[1-4])\b|basement(?: level)?\s*(\d)"


def _plant_below_ground(lines):
    """Critical equipment the submission places in a basement, each with its quote."""
    out, seen = [], set()
    for i, l in enumerate(lines):
        m = re.search(_BASEMENT, l, re.I)
        if not m or re.search(r"sump|parking|cafeteria|kitchen|mail|print|crack|paint|walls", l, re.I):
            continue
        level = (m.group(1) or f"B{m.group(2)}").upper()
        ctx = [l] + ([lines[i - 1]] if re.match(r"^-\s*locat", l, re.I) else []) + [_header(lines, i)]
        for pat, name in _PLANT:
            hit = next((c for c in ctx if re.search(pat, c, re.I)), None)
            if hit is not None:      # numbered units (generator unit 1 / 2) count separately; repeats do not
                unit = re.search(r"(?:unit|substation)\s*\d", " ".join(ctx), re.I)
                key = (name, level, unit.group(0).lower() if unit else "")
                if key not in seen:
                    seen.add(key)
                    out.append(dict(item=name, level=level, quote=l))
                break
    return out


def _flood_claims(lines):
    """The broker's own statements about flood risk (to set against the model)."""
    pat = r"(no flood|minimal|low to moderate flood|natural (?:protection|drainage)|flood potential)"
    return [l for l in lines if re.search(r"flood", l, re.I) and re.search(pat, l, re.I)][:4]


def _recommendation(lines):
    for i, l in enumerate(lines):
        if re.match(r"^RECOMMENDATION\s*:?\s*$", l, re.I):
            nxt = [x for x in lines[i + 1:i + 6]]
            k = next((j for j, x in enumerate(nxt) if x), None)
            if k is None:
                return None
            out = []
            for x in nxt[k:]:
                if not x:
                    break
                out.append(x)
            return " ".join(out)
        if re.match(r"^RECOMMENDATION\s*:\s*\S", l, re.I):
            return l
    return None


def extract(text, pages=None, call=None):
    """Facts from the submission. Returns {field: dict(value, quote, page, how)} plus list-valued extras."""
    pages = pages or [text]
    lines = _lines(text)
    f = {}
    for name, (_, pat, parse) in FIELDS.items():
        for l in lines:
            if l and re.search(pat, l, re.I):
                v = parse(l)
                if v is not None and v != "":
                    f[name] = dict(value=v, quote=l, page=_page(l, pages), how="rules")
                    break
    n_base = f["floors"]["value"][1] if "floors" in f else 0
    f["areas"] = _areas(lines, n_base)
    f["landmarks"] = _landmarks(lines)
    f["plant_below_ground"] = _plant_below_ground(lines)
    f["flood_claims"] = _flood_claims(lines)
    rec = _recommendation(lines)
    if rec:
        f["recommendation"] = dict(value=rec, quote=rec, page=_page(rec, pages), how="rules")
    f["contradictions"], f["rejected"] = [], []
    if call is not None:
        _llm_fill(f, text, pages, call)
    if "construction" in f:
        f["housing_class"] = _construction(f["construction"]["value"])
    return f


LLM_PROMPT = """You are reading a broker's insurance submission for a building in Nairobi.
For each field below that the document states, copy the ONE shortest line that states it, word for word, exactly as
written. Do not calculate, convert or summarise. Leave out fields the document does not state.
{fields}
Also list up to 5 pairs of statements in the document that contradict each other or cannot both be true. Copy both
statements word for word; "why" is a short reason with no numbers that are not in the two statements.
Reply with JSON only:
{{"fields": {{"<field>": "<exact line>"}}, "contradictions": [{{"a": "<exact line>", "b": "<exact line>", "why": "<reason>"}}]}}

DOCUMENT:
{text}"""


def _llm_fill(f, text, pages, call):
    """LLM finds lines for facts the rules missed; every quote is checked against the document, and values are
    parsed from the quote by code."""
    missing = [k for k in FIELDS if k not in f]
    spec = "\n".join(f"- {k}: {FIELDS[k][0]}" for k in missing) or "- (none - only look for contradictions)"
    try:
        raw = call(LLM_PROMPT.format(fields=spec, text=text[:30000]))
        b = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception:
        return
    nt = _norm(text)
    for k, q in (b.get("fields") or {}).items():
        if k not in missing or not isinstance(q, str):
            continue
        if _norm(q) not in nt:
            f["rejected"].append(dict(field=k, quote=q, reason="not found word for word in the document"))
            continue
        v = FIELDS[k][2](q)
        if v is None or v == "":
            f["rejected"].append(dict(field=k, quote=q, reason="no value could be read from the quote"))
            continue
        f[k] = dict(value=v, quote=q.strip(), page=_page(q, pages), how="ai")
    import agent
    for c in (b.get("contradictions") or [])[:5]:
        if not isinstance(c, dict):
            continue
        a, b_, why = str(c.get("a", "")), str(c.get("b", "")), str(c.get("why", ""))
        if a and b_ and _norm(a) in nt and _norm(b_) in nt and not agent.verify_numbers(why, a + " " + b_):
            f["contradictions"].append(dict(a=a.strip(), b=b_.strip(), why=why.strip(),
                                            pages=(_page(a, pages), _page(b_, pages))))
        else:
            f["rejected"].append(dict(field="contradiction", quote=a[:80], reason="quote or reason failed the check"))


# ---------------------------------------------------------------- 4. checks
def _km(a, b):
    dy = (b[0] - a[0]) * 111.32
    dx = (b[1] - a[1]) * 111.32 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy), (math.degrees(math.atan2(dx, dy)) + 360) % 360


def _compass(deg):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) // 45) % 8]


@lru_cache(maxsize=256)
def _gazetteer():
    p = os.path.join(HERE, "data", "gazetteer.csv")
    if not os.path.exists(p):
        return {}
    g = pd.read_csv(p).dropna(subset=["lat", "lon"])
    return {str(r.place_name).lower(): (float(r.lat), float(r.lon)) for r in g.itertuples()}


@lru_cache(maxsize=256)
def geocode_place(name):
    """A Nairobi place: the project gazetteer first, then OpenStreetMap (needs internet). None if not found."""
    key = name.strip().lower()
    if key in _gazetteer():
        return _gazetteer()[key]
    try:
        import geocode
        r = geocode.nominatim(f"{name}, Nairobi")
        if r and BBOX[0] <= r[0] <= BBOX[1] and BBOX[2] <= r[1] <= BBOX[3]:
            return float(r[0]), float(r[1])
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _waterways():
    p = os.path.join(HERE, "data", "osm", "rivers.json")
    if not os.path.exists(p):
        return None
    return np.asarray(json.load(open(p))["data"], dtype=float).reshape(-1, 2)


def nearest_place(lat, lon):
    g = _gazetteer()
    if not g:
        return None
    name, (la, lo) = min(g.items(), key=lambda kv: _km((lat, lon), kv[1])[0])
    return name.title().replace("Cbd", "CBD"), _km((lat, lon), (la, lo))[0]


def _flag(level, title, detail, quotes=(), ask=None):
    return dict(level=level, title=title, detail=detail, quotes=[q for q in quotes if q], ask=ask)


def checks(f, geocoder=geocode_place):
    """Rule-based checks of the submission's own figures. Each: level red / amber / info, title, detail, quotes."""
    out = []
    v = lambda k: f[k]["value"] if k in f else None
    q = lambda k: (f[k]["quote"], f[k].get("page")) if k in f else None
    ll = v("coords")

    # location: coordinates against the stated area, landmarks and waterways
    if ll and "address" in f and geocoder:
        for part in [p.strip() for p in v("address").split(",")]:
            if re.match(r"^(lot|block|plot|lr|l\.r|\d)", part, re.I) or part.lower() in ("nairobi", "kenya"):
                continue
            name = re.sub(r"\s+area$", "", part, flags=re.I)
            at = geocoder(name)
            if at:
                d, _ = _km(ll, at)
                if d > 1.2:
                    near = nearest_place(*ll)
                    out.append(_flag("red", "Coordinates don't match the address",
                                     f"The GPS point is {d:.1f} km from {name}"
                                     + (f"; the nearest named place is {near[0]} ({near[1]:.1f} km)." if near else "."),
                                     [q("address"), q("coords")], ask=f"Which is right - the address ({name}) or the "
                                                                      f"GPS coordinates? The two are {d:.1f} km apart."))
                break
    if ll and f.get("landmarks") and geocoder:
        bad = []
        for lm in f["landmarks"]:
            if re.search(r"river|stream", lm["name"], re.I):
                continue
            at = geocoder(re.sub(r"\s+area$", "", lm["name"], flags=re.I))
            if not at:
                continue
            d, brg = _km(ll, at)
            want = _bearing_of(lm["direction"])
            off = abs((brg - want + 180) % 360 - 180) if want is not None else 0
            if abs(d - lm["km"]) > max(1.0, 0.5 * lm["km"]) or off > 67.5:
                bad.append((lm, d, brg))
        if bad:
            out.append(_flag("red", f"{len(bad)} of {len(f['landmarks'])} landmark distances are wrong",
                             "; ".join(f"{lm['name']}: stated {lm['km']:g} km {lm['direction']}, actually "
                                       f"{d:.1f} km {_compass(brg)}" for lm, d, brg in bad) + " (from the GPS point).",
                             [lm["quote"] for lm, _, _ in bad]))
    if ll and "river" in f and _waterways() is not None:
        w = _waterways()
        dd = np.hypot((w[:, 0] - ll[0]) * 111.32, (w[:, 1] - ll[1]) * 111.32 * math.cos(math.radians(ll[0])))
        j = int(dd.argmin())
        near_km, brg = _km(ll, (w[j, 0], w[j, 1]))
        if v("river")["km"] > near_km + 0.5:
            out.append(_flag("amber", "A river or stream is closer than stated",
                             f"Stated: {v('river')['name']} {v('river')['km']:g} km away. The nearest mapped river or "
                             f"stream (OpenStreetMap) is {near_km:.1f} km {_compass(brg)} of the GPS point.", [q("river")]))

    # elevations: the drop to the river over its distance
    if v("elevation_m") and v("river_elevation_m") and "river" in f:
        drop = v("elevation_m") - v("river_elevation_m")
        slope = drop / max(v("river")["km"] * 1000, 1)
        if slope > 0.05:
            out.append(_flag("red", "Impossible height above the river",
                             f"A {drop:,.0f} m drop over {v('river')['km']:g} km is a {slope:.0%} average slope. "
                             f"The memo's 'natural protection' from flooding rests on this figure.",
                             [q("elevation_m"), q("river_elevation_m")],
                             ask="What survey supports the site and river elevations?"))

    # floor areas
    areas = f.get("areas") or []
    if areas and v("gfa_m2"):
        tot = sum(a["m2"] * a["n"] for a in areas)
        if abs(tot - v("gfa_m2")) / v("gfa_m2") > 0.05:
            out.append(_flag("red", "Floor areas don't add up",
                             f"The listed floors total {tot:,.0f} m² but the gross floor area is stated as "
                             f"{v('gfa_m2'):,.0f} m² ({(tot - v('gfa_m2')) / v('gfa_m2'):+.0%}).",
                             [q("gfa_m2")] + [(a["quote"], None) for a in areas],
                             ask="Please provide the measured floor areas (the breakdown and the total disagree)."))
    if v("height_m") and v("storey_m") and "floors" in f:
        above = v("floors")[0]
        est = (v("ground_storey_m") or v("storey_m")) + (above - 1) * v("storey_m")
        if abs(est - v("height_m")) / v("height_m") > 0.05:
            out.append(_flag("amber", "Height doesn't match the floors",
                             f"{above} floors at the stated storey heights make about {est:.1f} m, not "
                             f"{v('height_m'):g} m.", [q("height_m"), q("storey_m")]))

    # services
    if v("water_daily_l") and v("water_monthly_m3"):
        monthly_from_daily = v("water_daily_l") * 30 / 1000
        r = v("water_monthly_m3") / monthly_from_daily
        if r > 3 or r < 1 / 3:
            out.append(_flag("amber", "Water figures disagree",
                             f"{v('water_daily_l'):,.0f} litres a day is about {monthly_from_daily:,.0f} m³ a month, "
                             f"but {v('water_monthly_m3'):,.0f} m³ a month is stated ({r:,.0f}× more).",
                             [q("water_daily_l"), q("water_monthly_m3")]))
    if v("fire_tank_l") and v("sprinkler_m3h"):
        mins = v("fire_tank_l") / 1000 / v("sprinkler_m3h") * 60
        if mins < 30:
            out.append(_flag("amber", "Fire water runs out fast",
                             f"A {v('fire_tank_l'):,.0f} litre tank feeds sprinklers rated {v('sprinkler_m3h'):,.0f} m³ "
                             f"an hour for about {mins:.1f} minutes.", [q("fire_tank_l"), q("sprinkler_m3h")]))
    base_m2 = sum(a["m2"] * a["n"] for a in areas if a["kind"] == "basement")
    if v("sump_m3h") and base_m2:
        hrs = base_m2 * 0.10 / v("sump_m3h")
        if hrs > 2:
            out.append(_flag("amber", "Basement pumps are small",
                             f"Pumping out 10 cm of water from {base_m2:,.0f} m² of basement at {v('sump_m3h'):g} m³ "
                             f"an hour takes about {hrs:.0f} hours.", [q("sump_m3h")],
                             ask="What stops surface water entering the basements (ramp barriers, flood doors)?"))

    # what sits below ground
    plant = f.get("plant_below_ground") or []
    if plant:
        items = pd.DataFrame(plant).groupby(["item", "level"]).size().reset_index(name="n")
        txt = ", ".join(f"{r.item.lower()}{'s' if r.n > 1 else ''} ({r.level})" for r in items.itertuples())
        out.append(_flag("red", "Critical plant is below ground",
                         f"{txt}. A flooded basement would cut power, cooling and water to the whole building; "
                         "business interruption is not priced here.", [(p["quote"], None) for p in plant[:6]],
                         ask="Can the basement plant be protected or raised, and is business interruption included?"))

    # the broker's role and timing
    if "recommendation" in f and re.search(r"\b(accept|standard rates?)\b", v("recommendation"), re.I):
        out.append(_flag("amber", "The broker recommends the terms",
                         "The submission includes its own underwriting recommendation; treat it as the broker's view.",
                         [q("recommendation")]))
    if v("issued") and v("expiry"):
        days = (v("expiry") - v("issued")).days
        if 0 <= days <= 7:
            out.append(_flag("amber", f"Only {days} days to respond",
                             f"Issued {v('issued'):%d %b %Y}, expires {v('expiry'):%d %b %Y}.", [q("issued"), q("expiry")]))
    if v("tiv_kes") and v("gfa_m2"):
        out.append(_flag("info", "Value per square metre",
                         f"KES {v('tiv_kes') / v('gfa_m2'):,.0f} per m² of stated floor area - check against a "
                         "quantity surveyor's rebuild valuation.", [q("tiv_kes")]))
    for c in f.get("contradictions") or []:
        out.append(_flag("amber", "Statements contradict each other (AI-found)", c["why"],
                         [(c["a"], c["pages"][0]), (c["b"], c["pages"][1])]))
    for x in out:
        x["quotes"] = [(qq, None) if isinstance(qq, str) else qq for qq in x["quotes"] if qq]
    return out


# ---------------------------------------------------------------- 5. building shape
def value_shares(f):
    """Share of the building's value on the ground floor and in basements (by floor area). None if unknown."""
    above, base = f["floors"]["value"] if "floors" in f else (None, 0)
    areas = f.get("areas") or []
    if areas:
        tot = sum(a["m2"] * a["n"] for a in areas)
        g = sum(a["m2"] * a["n"] for a in areas if a["kind"] == "ground")
        b = sum(a["m2"] * a["n"] for a in areas if a["kind"] == "basement")
        if g == 0 and above:
            g = (tot - b) / above
        return dict(ground=g / tot, basement=b / tot, upper=1 - (g + b) / tot, basis="floor areas in the submission")
    if above and (above >= 3 or base):
        n = above + base
        return dict(ground=1 / n, basement=base / n, upper=(above - 1) / n, basis="floor count (equal floors ASSUMED)")
    return None


def building_events(events, cls, tiv, shares, fill_m=None):
    """Event table for a multi-storey building: the ground floor takes the street depth, the basements fill once
    water reaches the street (ASSUMED), upper floors stay dry. Returns (events, aal)."""
    p = cm.VULN[cls]
    e = events.copy()
    dr_g = cm.damage_ratio(e.depth_m.to_numpy(), p["depth_factor"], p["cap"])
    dr_b = np.where(e.depth_m.to_numpy() >= BASEMENT_TRIGGER_M,
                    cm.damage_ratio(fill_m or BASEMENT_FILL_M, p["depth_factor"], p["cap"]), 0.0)
    e["damage_pct"] = (shares["ground"] * dr_g + shares["basement"] * dr_b) * 100
    e["loss_kes"] = e.damage_pct / 100 * tiv
    aal = float(cm.aal_from_ep(e.return_period.to_numpy(float), e.loss_kes.to_numpy()))
    return e, aal


def apply(o, f, flags, source_name=""):
    """Put the submission into an evaluation: building-shape losses (all cards then agree), the checks, and the
    broker-says / model-says comparison. Returns o (modified)."""
    import financial as fin
    shares = value_shares(f)
    if shares and (shares["basement"] > 0 or shares["upper"] > 0.3):
        e0, aal0 = o["events"], o["aal_kes"]
        e, aal = building_events(e0, o["housing_class"], o["tiv_kes"], shares)
        ded, lim = fin.policy_terms([o["tiv_kes"]])
        e["insured_loss_kes"] = fin.insured_loss(e.loss_kes.to_numpy()[None], ded, lim)[0]
        rps = e.return_period.to_numpy(float)
        o.update(events_block=e0, aal_block=aal0, events=e, aal_kes=aal, rate_per_mille=aal / o["tiv_kes"] * 1000,
                 aal_insured_kes=float(cm.aal_from_ep(rps, e.insured_loss_kes.to_numpy())), shares=shares)
        if "port_100_before" in o:
            j = list(e.return_period).index(100) if 100 in set(e.return_period) else len(e) // 2
            o["port_100_after"] = o["port_100_before"] + float(e.loss_kes.iloc[j])
        r100 = e.set_index("return_period").loc[100] if 100 in set(e.return_period) else e.iloc[len(e) // 2]
        o["steps"].append(dict(title="Building shape (from the submission)", text=(
            f"Value by floor ({shares['basis']}): basements {shares['basement']:.0%}, ground floor "
            f"{shares['ground']:.0%}, upper floors {shares['upper']:.0%}. Water at street level damages the ground "
            f"floor (same damage curve) and is ASSUMED to flood the basements about {BASEMENT_FILL_M:g} m deep once "
            f"it reaches {BASEMENT_TRIGGER_M * 100:.0f} cm; upper floors stay dry. Treating the building as one block "
            f"gave KES {aal0:,.0f} a year; by floor it is **KES {aal:,.0f}** a year, and a 1-in-100 flood costs "
            f"KES {r100.loss_kes / 1e6:,.1f} m.")))
    # insured figures under the submission's own flood deductible, when it states one
    if "flood_deductible" in f:
        pct, mn = f["flood_deductible"]["value"]
        ded = min(max(pct / 100 * o["tiv_kes"], mn), o["tiv_kes"])
        e = o["events"]
        e["insured_loss_kes"] = fin.insured_loss(e.loss_kes.to_numpy()[None], [ded], [o["tiv_kes"]])[0]
        o.update(deductible_kes=ded, deductible_source="submission",
                 aal_insured_kes=float(cm.aal_from_ep(e.return_period.to_numpy(float), e.insured_loss_kes.to_numpy())))
    # a GPS point that contradicts the address: the location itself is in doubt
    if any(x["title"].startswith("Coordinates") for x in flags):
        o["confidence"] = "Low"
        o["confidence_reasons"] = ["the submission's GPS point and address disagree"] + o["confidence_reasons"]
    # broker says vs model says
    rows = []
    if f.get("flood_claims"):
        bad = o["risk_band"] in ("High", "Very high")
        strongest = sorted(f["flood_claims"], key=lambda c: next((i for i, w in enumerate(
            ("minimal", "natural", "low", "no flood")) if w in c.lower()), 9))[0]
        rows.append(dict(topic="Flood risk", broker=strongest.lstrip("-✓ "),
                         model=f"{o['risk_band']} - more flood-prone than {o['city_percentile']:.0f}% of Nairobi",
                         tone="bad" if bad else "good"))
    ev_ = o.get("ai_evidence")
    if ev_ is not None and len(ev_):
        hist = next((c for c in f.get("flood_claims") or [] if re.search(r"no flood", c, re.I)), None)
        if hist:
            seen = {}
            for n in ev_.place_name.head(4):
                seen.setdefault(n.lower(), n)
            names = ", ".join(seen.values())
            rows.append(dict(topic="Flood history", broker=hist,
                             model=f"Flood reports name {names}, {ev_.distance_km.min():.1f} km away", tone="warn"))
    loc = next((x for x in flags if x["title"].startswith("Coordinates")), None)
    if loc and "address" in f:
        rows.append(dict(topic="Location", broker=f["address"]["value"], model=loc["detail"], tone="bad"))
    if "flood_deductible" in f and "tiv_kes" in f:
        pct, mn = f["flood_deductible"]["value"]
        ded = max(pct / 100 * o["tiv_kes"], mn)
        e = o["events"][o["events"].loss_kes > 0]
        if len(e):
            r0 = e.iloc[0]
            share = min(ded / r0.loss_kes, 1)
            sent = next((x for x in re.split(r"(?<=\.)\s+", f["flood_deductible"]["quote"]) if "%" in x),
                        f["flood_deductible"]["quote"])
            rows.append(dict(topic="Flood deductible", broker=sent,
                             model=f"KES {ded / 1e6:,.1f} m - the insured keeps {share:.0%} of a 1-in-"
                                   f"{int(r0.return_period)} flood loss (KES {r0.loss_kes / 1e6:,.0f} m)",
                             tone="warn" if share >= 0.5 else "good"))
    o["submission"] = dict(source=source_name, flags=flags, compare=rows, facts=facts_table(f), raw=f,
                           n_facts=sum(1 for k in FIELDS if k in f), rejected=f.get("rejected", []),
                           client=(f.get("client") or {}).get("value"))
    return o


def facts_table(f):
    """What was read, for display: fact, value, the quote it came from, page, how it was found."""
    rows = []
    for k, (desc, _, _) in FIELDS.items():
        if k in f:
            val = f[k]["value"]
            if k == "coords":
                val = f"{val[0]:.5f}, {val[1]:.5f}"
            elif k == "floors":
                val = f"{val[0]} above ground + {val[1]} basement"
            elif k == "river":
                val = f"{val['name']} {val['km']:g} km {val['direction'] or ''}".strip()
            elif k == "flood_deductible":
                val = f"{val[0]:g}% (minimum KES {val[1]:,.0f})"
            elif isinstance(val, float):
                val = f"{val:,.0f}" if val >= 100 else f"{val:g}"
            rows.append(dict(fact=desc, value=str(val), quote=f[k]["quote"], page=f[k].get("page"),
                             found_by="AI (quote checked)" if f[k]["how"] == "ai" else "rules"))
    return pd.DataFrame(rows)
