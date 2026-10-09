"""Quote requests: a building owner who used the public estimate page asks for cover; the request lands in the
underwriters' queue on the Evaluate page, which evaluates it in one click; the underwriter's decision closes it.

Stored on this computer only (regions/nairobi/quote_requests.csv, git-ignored) - a prototype, not a production intake:
contact details are kept for the underwriter to call back, never sent anywhere by the system.
"""
import datetime
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "regions", "nairobi", "quote_requests.csv")
COLUMNS = ["id", "at", "name", "contact", "place", "lat", "lon", "housing_class", "tiv_kes", "estimate_aal_kes",
           "risk_band", "status", "decision_id"]


def load():
    try:
        return pd.read_csv(PATH, dtype={"contact": str, "status": str, "decision_id": str}).astype(  # text columns stay
            {"decision_id": object, "status": object}) if os.path.exists(PATH) and os.path.getsize(PATH) else \
            pd.DataFrame(columns=COLUMNS)                                                             # text (pandas 3)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)


def add(o, name, contact):
    """o: the public page's evaluation. Returns the request reference, e.g. Q-0003."""
    q = load()
    row = dict(id=f"Q-{len(q) + 1:04d}", at=datetime.datetime.now().isoformat(timespec="seconds"), name=name.strip(),
               contact=contact.strip(), place=o["label"], lat=o["lat"], lon=o["lon"], housing_class=o["housing_class"],
               tiv_kes=o["tiv_kes"], estimate_aal_kes=o["aal_kes"], risk_band=o["risk_band"], status="waiting",
               decision_id="")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    pd.concat([q, pd.DataFrame([row])], ignore_index=True)[COLUMNS].to_csv(PATH, index=False)
    return row["id"]


def waiting():
    q = load()
    return q[q.status == "waiting"] if len(q) else q


def close(request_id, decision, decision_id):
    q = load()
    q.loc[q.id == request_id, ["status", "decision_id"]] = [f"decided: {decision}", decision_id]
    q.to_csv(PATH, index=False)
