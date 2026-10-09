"""Underwriting decisions: what the underwriter decided on an evaluation, and the risks written into the book.

The model suggests a stance (briefing.stance - rules, never the LLM); the underwriter decides. Each decision is logged
with the suggestion beside it, so overrides are visible. 'Accept' and 'Accept with loading' add the risk to the
portfolio, so the Flood briefing, Accumulation and Reinsurance pages include it; claim decisions are logged only.
Nothing is deleted: a decision can be withdrawn, which keeps the audit trail.

Stored in the database (store.py, table 'decisions', one 'book' per portfolio); every decision is also a line in the
audit trail (table 'events').
"""
import datetime
import os

import numpy as np
import pandas as pd

import store

HERE = os.path.dirname(os.path.abspath(__file__))
RISK_DECISIONS = ["Accept", "Accept with loading", "Refer", "Decline"]
CLAIM_DECISIONS = ["Pay", "Send a loss adjuster", "Query the insured", "Decline"]
WRITES = {"Accept", "Accept with loading"}
COLUMNS = ["id", "at", "kind", "label", "lat", "lon", "housing_class", "tiv_kes", "decision", "loading_pct", "note",
           "suggested", "risk_band", "site_score", "final_score", "technical_premium_kes", "quoted_premium_kes",
           "claimed_kes", "port_100_before_kes", "port_100_after_kes", "source", "status"]


def load(book="nairobi"):
    rows = store.decisions(book)
    d = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("lat", "lon", "tiv_kes", "loading_pct", "site_score", "final_score", "technical_premium_kes",
              "quoted_premium_kes", "claimed_kes", "port_100_before_kes", "port_100_after_kes"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
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
    superseded = list(_same(log, o).id)                 # a new decision on the same risk supersedes the old one
    log.loc[log.id.isin(superseded), "status"] = "superseded"
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
    store.put_decisions(book, log[log.id.isin(superseded)].to_dict("records") + [row])
    store.log(book, f"{row['id']} {'claim' if is_claim else 'risk'} '{row['label']}': {decision}"
              + (f" (+{load_pct:.0f}%)" if load_pct else "") + f"; model suggested '{suggested}'"
              + (f"; note: {note}" if note else ""), kind="decision")
    if is_claim:                          # a paid claim teaches the model; a changed mind takes the evidence back
        for old_id in superseded:
            claim_evidence(old_id, book, remove=True)
        if decision == "Pay":
            row["evidence_added"] = claim_evidence(row, book)
    return row


def withdraw(decision_id, book="nairobi"):
    log = load(book)
    log.loc[log.id == decision_id, "status"] = "withdrawn"
    store.put_decisions(book, log[log.id == decision_id].to_dict("records"))
    store.log(book, f"{decision_id} withdrawn", kind="decision")
    claim_evidence(decision_id, book, remove=True)


def _region_key(book):
    return "nairobi" if book == "nairobi" else book[len("region-"):] if book.startswith("region-") else None


def claim_evidence(row, book="nairobi", remove=False):
    """A PAID flood claim is confirmed flooding at that spot - the best evidence there is. It becomes an approved flood
    report in the book's region (Model workspace), so the evidence layer raises nearby prices at once and the next
    model trained there learns from it. Withdrawing the decision removes it. Returns True if evidence changed."""
    import workspace as ws
    key = _region_key(book)
    if key is None:                       # an uploaded book outside the workspace has no region to learn into
        return False
    r = ws.Region(key)
    cid = "claim-" + (row if isinstance(row, str) else row["id"])
    cands = [c for c in r.candidates() if c["id"] != cid]
    if not remove:
        ratio = row["claimed_kes"] / row["tiv_kes"] if row["tiv_kes"] else 0
        quote = (f"Flood claim {row['id']} paid: KES {row['claimed_kes']:,.0f} ({ratio:.0%} of the insured value) at "
                 f"{row['label']}." + (f" Note: {row['note']}" if isinstance(row.get("note"), str) and row["note"] else ""))
        sig = dict(place_name=row["label"], place_type="other", mechanism="unknown",
                   severity=3 if ratio >= 0.3 else 2 if ratio >= 0.1 else 1, event_date=str(row["at"])[:7],
                   evidence_quote=quote, confidence=1.0, source_id=row["id"],
                   source_title=f"Paid flood claim {row['id']} (decision log)", source_url="", source_date=row["at"][:10])
        cands.append(dict(id=cid, place_name=row["label"], place_type="other", mechanism="unknown",
                          severity=sig["severity"], confidence=1.0, n_sources=1, lat=float(row["lat"]),
                          lon=float(row["lon"]), found_as="claim location", status="approved", signals=[sig]))
    elif len(cands) == len(r.candidates()):
        return False
    r.save_candidates(cands)
    ws.review(r, {})                      # rewrites the region's approved evidence (signals.csv)
    return True


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
