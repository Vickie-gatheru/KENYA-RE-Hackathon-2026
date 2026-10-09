"""Quote requests: a building owner who used the public estimate page asks for cover; the request lands in the
underwriters' queue on the Evaluate page, which evaluates it in one click; the underwriter's decision closes it.

Stored in the database (store.py, table 'quote_requests'; each request and its closing is also in the audit trail) on
this computer only - a prototype, not a production intake: contact details are kept for the underwriter to call back,
never sent anywhere by the system.
"""
import datetime

import pandas as pd

import store

COLUMNS = ["id", "at", "name", "contact", "place", "lat", "lon", "housing_class", "tiv_kes", "estimate_aal_kes",
           "risk_band", "status", "decision_id"]


def load():
    q = pd.DataFrame(store.quote_requests(), columns=COLUMNS)
    for c in ("lat", "lon", "tiv_kes", "estimate_aal_kes"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    return q.astype({"id": object, "contact": object, "status": object, "decision_id": object})


def add(o, name, contact):
    """o: the public page's evaluation. Returns the request reference, e.g. Q-0003."""
    row = dict(id=f"Q-{len(load()) + 1:04d}", at=datetime.datetime.now().isoformat(timespec="seconds"),
               name=name.strip(), contact=contact.strip(), place=o["label"], lat=float(o["lat"]), lon=float(o["lon"]),
               housing_class=o["housing_class"], tiv_kes=float(o["tiv_kes"]), estimate_aal_kes=float(o["aal_kes"]),
               risk_band=o["risk_band"], status="waiting", decision_id="")
    store.put_quote_request(row)
    store.log("public", f"quote request {row['id']} for {row['place']} (estimate KES {row['estimate_aal_kes']:,.0f}/yr)",
              kind="quote request")
    return row["id"]


def waiting():
    q = load()
    return q[q.status == "waiting"] if len(q) else q


def close(request_id, decision, decision_id):
    for row in store.quote_requests():
        if row["id"] == request_id:
            row.update(status=f"decided: {decision}", decision_id=decision_id)
            store.put_quote_request(row)
            store.log("public", f"quote request {request_id} closed: {decision} ({decision_id})", kind="quote request")
