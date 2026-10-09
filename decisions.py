"""Underwriting decisions: what the underwriter decided on an evaluation, and the risks written into the book.

The model suggests a stance (briefing.stance - rules, never the LLM); the underwriter decides. Each decision is logged
with the suggestion beside it, so overrides are visible. 'Accept' and 'Accept with loading' add the risk to the
portfolio, so the Flood briefing, Accumulation and Reinsurance pages include it; claim decisions are logged only.
Nothing is deleted: a decision can be withdrawn, which keeps the audit trail.

Stored per book in regions/<book>/decisions.csv (git-ignored, like the rest of regions/).
"""
import datetime
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "regions")
RISK_DECISIONS = ["Accept", "Accept with loading", "Refer", "Decline"]
CLAIM_DECISIONS = ["Pay", "Send a loss adjuster", "Query the insured", "Decline"]
WRITES = {"Accept", "Accept with loading"}
COLUMNS = ["id", "at", "kind", "label", "lat", "lon", "housing_class", "tiv_kes", "decision", "loading_pct", "note",
           "suggested", "risk_band", "site_score", "final_score", "technical_premium_kes", "quoted_premium_kes",
           "claimed_kes", "port_100_before_kes", "port_100_after_kes", "source", "status"]


def path(book="nairobi"):
    return os.path.join(ROOT, book, "decisions.csv")


def load(book="nairobi"):
    p = path(book)
    try:
        d = pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 0 else pd.DataFrame(columns=COLUMNS)
    except pd.errors.EmptyDataError:
        d = pd.DataFrame(columns=COLUMNS)
    return d


def default_for(suggested, is_claim=False):
    """The option the bar starts on, from the rule-based suggestion."""
    s = (suggested or "").lower()
    if is_claim:
        return "Pay" if "consistent" in s else "Send a loss adjuster"
    if s.startswith("standard"):
        return "Accept"
    if "loading" in s:
        return "Accept with loading"
    return "Refer"


def _same(log, o):
    if not len(log):
        return log.iloc[0:0]
    m = ((log.status == "active") & (log.label == o["label"]) & np.isclose(log.lat.astype(float), o["lat"])
         & np.isclose(log.lon.astype(float), o["lon"]) & np.isclose(log.tiv_kes.astype(float), o["tiv_kes"]))
    return log[m]


def existing(o, book="nairobi"):
    """The active decision already recorded for this evaluation, or None."""
    s = _same(load(book), o)
    return s.iloc[-1].to_dict() if len(s) else None


def record(o, decision, suggested, note="", loading_pct=0.0, book="nairobi", source="form"):
    """Log a decision (replacing an earlier active one for the same evaluation). Returns the new row."""
    log = load(book)
    is_claim = bool(o.get("claim"))
    for i in _same(log, o).index:                       # a new decision on the same risk supersedes the old one
        log.loc[i, "status"] = "superseded"
    n = len(log) + 1
    load_pct = float(loading_pct or 0) if decision == "Accept with loading" else 0.0
    row = dict(id=f"{'C' if is_claim else 'W'}-{n:04d}", at=datetime.datetime.now().isoformat(timespec="seconds"),
               kind="claim" if is_claim else "risk", label=o["label"], lat=o["lat"], lon=o["lon"],
               housing_class=o["housing_class"], tiv_kes=o["tiv_kes"], decision=decision, loading_pct=load_pct,
               note=note, suggested=suggested, risk_band=o["risk_band"], site_score=o["site_score"],
               final_score=o["final_score"], technical_premium_kes=o["aal_kes"],
               quoted_premium_kes=o["aal_kes"] * (1 + load_pct / 100) if decision in WRITES else np.nan,
               claimed_kes=(o.get("claim") or {}).get("claimed_kes", np.nan),
               port_100_before_kes=o.get("port_100_before", np.nan), port_100_after_kes=o.get("port_100_after", np.nan),
               source=source, status="active")
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)[COLUMNS]
    os.makedirs(os.path.dirname(path(book)), exist_ok=True)
    log.to_csv(path(book), index=False)
    return row


def withdraw(decision_id, book="nairobi"):
    log = load(book)
    log.loc[log.id == decision_id, "status"] = "withdrawn"
    log.to_csv(path(book), index=False)


def written(book="nairobi"):
    log = load(book)
    if not len(log):
        return log
    return log[(log.status == "active") & (log.kind == "risk") & log.decision.isin(WRITES)]


def with_written(d, book="nairobi"):
    """The portfolio plus the risks written here. A written risk carries the flood-map score at its site, like every
    portfolio building (the AI layer is applied on top, as for the rest of the book)."""
    w = written(book)
    if not len(w):
        return d
    add = pd.DataFrame({"loc_id": w.id.values, "lat": w.lat.astype(float).values, "lon": w.lon.astype(float).values,
                        "housing_class": w.housing_class.values, "tiv_kes": w.tiv_kes.astype(float).values,
                        "hazard_score_common": w.site_score.astype(float).clip(0, 1).values})
    add["written_here"] = True
    out = pd.concat([d.assign(written_here=False) if "written_here" not in d else d, add], ignore_index=True)
    return out
