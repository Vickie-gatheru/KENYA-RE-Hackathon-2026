"""Load any insurer's building list (CSV or Excel) into the model - its own column names, free-text building types,
values only - with every row it cannot use rejected and explained, never passed on as NaN.

  read_table(data, name)          bytes or path -> DataFrame (csv / xlsx / xls)
  guess_columns(df)               {model field: the file's column or None}, from common bordereau names
  map_classes(values)             building-type text -> model class, with a review table of what mapped to what
  prepare(df, colmap, overrides)  -> (portfolio ready for catmodel, report)

The portfolio gets: loc_id, lat, lon, housing_class, tiv_kes, hazard_score_common (read from the flood map at each
building - the 'common' tier only), plus floor_area_m2 / floors when the file has them.
"""
import io
import os
import re

import numpy as np
import pandas as pd

import catmodel as cm
import underwriting as uw

FIELDS = {   # model field: (what it is, common names in bordereaux, required)
    "loc_id": ("building / policy reference", ["loc_id", "location id", "risk id", "policy ref", "policyref",
                                                "policy no", "policy number", "policy", "reference", "ref", "id"], False),
    "lat": ("latitude", ["lat", "latitude", "gps lat", "y coord", "y"], True),
    "lon": ("longitude", ["lon", "lng", "long", "longitude", "gps lon", "x coord", "x"], True),
    "tiv_kes": ("insured value (KES)", ["tiv", "tiv kes", "sum insured", "suminsured", "tsi", "total sum insured",
                                         "insured value", "total insured value", "building value", "value"], True),
    "housing_class": ("building type", ["housing class", "construction", "construction type", "building type",
                                         "construction class", "class", "type", "structure"], True),
    "floor_area_m2": ("floor area (m²)", ["floor area m2", "floor area", "area m2", "gfa", "area"], False),
    "cost_per_m2_kes": ("rebuild cost per m² (KES)", ["cost per m2 kes", "cost per m2", "rebuild cost per m2",
                                                       "cost m2"], False),
    "floors": ("number of floors", ["floors", "storeys", "stories", "no of floors", "number of floors"], False),
    "hazard_score_common": ("flood score 0-1 (if the file has one)", ["hazard score common", "flood score",
                                                                       "hazard score"], False),
}
# extra building-type words the bordereaux use, on top of underwriting.SYNONYMS (checked first, so
# 'concrete block' is masonry - block walls - not a reinforced-concrete frame)
EXTRA_WORDS = [("permanent_masonry", ["concrete block", "block wall", "brick", "stone", "masonry", "blocks"]),
               ("concrete_rcc", ["rc frame", "r.c.", "rcc", "reinforced", "concrete frame", "steel frame", "high rise",
                                 "multi-storey", "multi storey"]),
               ("informal_iron_sheet", ["mabati", "iron sheet", "corrugated", "informal", "shack"]),
               ("semi_permanent", ["timber", "mud", "wattle", "semi permanent", "semi-permanent", "wood"])]


def _key(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def read_table(data, name="upload.csv"):
    if isinstance(data, str) and os.path.exists(data):
        name, data = data, open(data, "rb").read()
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else io.StringIO(data))


def guess_columns(df):
    """Best guess for each model field: exact name match first, then a name that contains a known word."""
    cols = {_key(c): c for c in df.columns}
    out, used = {}, set()
    for field, (_, names, _) in FIELDS.items():
        names = [_key(n) for n in names]
        hit = next((cols[n] for n in names if n in cols and cols[n] not in used), None)
        if hit is None:
            hit = next((c for k, c in cols.items() for n in names if len(n) > 2 and n in k and c not in used), None)
        out[field] = hit
        if hit is not None:
            used.add(hit)
    return out


def classify(text):
    t = _key(text)
    if not t:
        return None
    for cls in cm.VULN:
        if t == cls.replace("_", " "):
            return cls
    for cls, words in EXTRA_WORDS:
        if any(w in t for w in words):
            return cls
    return uw.normalise_class(t)


def map_classes(values):
    """Review table: each distinct building-type text, how many rows, and the class it maps to (None = unknown)."""
    vc = pd.Series(values).fillna("(blank)").astype(str).value_counts()
    return pd.DataFrame({"building type in file": vc.index, "rows": vc.values,
                         "model type": [classify(v) if v != "(blank)" else None for v in vc.index]})


def prepare(df, colmap=None, class_overrides=None, default_class=None, sample_map=True, bbox=None):
    """Clean portfolio + report. class_overrides: {file text: model class} from the review table. default_class:
    used for unknown types if given; otherwise those rows are rejected. A flood-score column in the file is used as
    given (the organisers' format); otherwise each building's score is read from the flood map (sample_map). bbox =
    (lat_min, lat_max, lon_min, lon_max) of the region, for the missing-minus-sign fix (default Nairobi)."""
    import hazard as hz
    colmap = colmap or guess_columns(df)
    missing = [FIELDS[f][0] for f, (_, _, req) in FIELDS.items() if req and not colmap.get(f)]
    if missing:
        raise ValueError(f"Choose the column for: {', '.join(missing)}")
    d = pd.DataFrame({f: df[c] for f, c in colmap.items() if c})
    fixes, reasons = [], pd.Series("", index=d.index)

    for c in ("lat", "lon", "tiv_kes", "floor_area_m2", "cost_per_m2_kes", "floors", "hazard_score_common"):
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(r"[,\s]|KES|KSh", "", regex=True)
                                 .replace({"": np.nan, "nan": np.nan}), errors="coerce")
    # Nairobi is south of the equator: a positive latitude that lands on the map when negated lost its sign
    lat0, lat1, lon0, lon1 = bbox or (-1.45, -1.10, 36.60, 37.10)
    flip = (d.lat > 0) & (-d.lat).between(lat0, lat1) & d.lon.between(lon0, lon1)
    if flip.any():
        d.loc[flip, "lat"] = -d.loc[flip, "lat"]
        fixes.append(f"{int(flip.sum())} latitudes had no minus sign (Nairobi is south of the equator) - fixed")
    # the starter kit's documented data fix (catmodel.load_exposure): insured value = floor area x cost per m2,
    # rounded to KES 5,000 - its tiv_kes column is about 10x that. Applied only when both columns exist and disagree.
    if {"floor_area_m2", "cost_per_m2_kes"} <= set(d.columns):
        calc = (d.floor_area_m2 * d.cost_per_m2_kes / 5000).round() * 5000
        ok_calc = calc.notna() & (calc > 0)
        off = ok_calc & ((d.tiv_kes - calc).abs() > 0.05 * calc)
        if off.mean() > 0.5:
            d.loc[ok_calc, "tiv_kes"] = calc[ok_calc]
            fixes.append(f"insured value recomputed as floor area x cost per m2 for {int(ok_calc.sum())} rows (the "
                         f"file's value column disagreed for {int(off.sum())}, as in the starter kit)")
    if "loc_id" not in d or d.loc_id.isna().all():
        d["loc_id"] = [f"ROW-{i + 2:05d}" for i in range(len(d))]       # +2: the spreadsheet row (header is row 1)
        fixes.append("no reference column - rows numbered as in the file")

    text = d.housing_class.fillna("").astype(str)
    over = {str(k): v for k, v in (class_overrides or {}).items() if v}
    d["housing_class"] = [over.get(t) or classify(t) or default_class for t in text]

    reasons[d.lat.isna() | d.lon.isna()] = "no usable coordinates"
    given = "hazard_score_common" in d and d.hazard_score_common.notna().any()
    if given:
        reasons[(reasons == "") & ~d.hazard_score_common.between(0, 1)] = "flood score missing or not between 0 and 1"
        fixes.append("flood scores taken from the file")
    elif sample_map:
        inside = hz.sample(d.lat.fillna(0).to_numpy(), d.lon.fillna(0).to_numpy())
        reasons[(reasons == "") & ~inside["inside"]] = "outside the flood map"
        d["hazard_score_common"] = inside["common"].astype(float)
    else:
        raise ValueError("No flood map for this region and no flood-score column in the file - add one or the other.")
    reasons[(reasons == "") & (d.tiv_kes.isna() | (d.tiv_kes <= 0))] = "no insured value"
    reasons[(reasons == "") & d.housing_class.isna()] = "building type not recognised"

    rejected = df.loc[reasons != ""].assign(reason=reasons[reasons != ""])
    ok = d.loc[reasons == ""].reset_index(drop=True)
    keep = ["loc_id", "lat", "lon", "housing_class", "tiv_kes", "hazard_score_common"] + \
           [c for c in ("floor_area_m2", "cost_per_m2_kes", "floors") if c in ok]
    ok = ok[keep]
    if len(ok):
        cm.check_exposure(ok)
    report = dict(rows_in=len(df), rows_used=len(ok), rejected=rejected, fixes=fixes,
                  reasons=rejected.reason.value_counts().to_dict(), columns=colmap,
                  tiv_kes=float(ok.tiv_kes.sum()) if len(ok) else 0.0,
                  share_on_flood_map=float((ok.hazard_score_common > 0).mean()) if len(ok) else 0.0)
    return ok, report


def summary_text(rep):
    parts = [f"{rep['rows_used']:,} of {rep['rows_in']:,} rows loaded (KES {rep['tiv_kes'] / 1e9:,.2f} bn insured, "
             f"{rep['share_on_flood_map']:.0%} on the flood map)"]
    if rep["reasons"]:
        parts.append("rejected: " + ", ".join(f"{n} {r}" for r, n in rep["reasons"].items()))
    return "; ".join(parts + rep["fixes"]) + "."
