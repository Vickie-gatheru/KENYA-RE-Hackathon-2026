"""Financial engine: ground-up loss -> insured loss (policy terms) -> reinsured loss (treaty), per flood event.

    ground-up (per building)      = damage ratio x insured value                         (catmodel.py)
    insured / gross (per building) = min(max(ground-up - deductible, 0), limit)
    policyholder retains           = ground-up - gross            (deductible + anything above the limit)
    treaty, on the cedant's gross portfolio loss G in each event:
        quota share ceded          = q x G
        cedant retained after QS   R = (1 - q) x G
        cat XL ceded               = min(max(R - retention, 0), limit)      ("limit xs retention", per event)
        reinsurer (Kenya Re) pays  = QS ceded + XL ceded
        cedant net                 = R - XL ceded

ALL TERMS ARE ASSUMED. The organisers' exposure file has no deductibles, limits or treaty, so the defaults below are
illustrative round numbers scaled to this SYNTHETIC book (layer attaches between the 1-in-10 and 1-in-25 events and
is exhausted just beyond 1-in-100). They are inputs to change, not market terms.
Simplifications: every building is insured (100% take-up); one flood per year (occurrence = aggregate EP); no
reinstatement premiums, no event-limit on the quota share, no per-risk surplus treaty.
Each event's financial loss is a monotone function of its ground-up loss, so applying terms to each return-period
scenario gives the financial EP curve directly; AAL uses the same trapezoid rule (a lower estimate).
"""
import numpy as np
import pandas as pd

import catmodel as cm

# ---------------------------------------------------------------- ASSUMED terms (edit + test sensitivity)
DED_PCT = 0.02             # deductible: 2% of the building's insured value ...
DED_MIN_KES = 10_000       # ... but at least KES 10,000 (never more than the value itself)
LIMIT_PCT = 1.0            # policy limit as a share of insured value (1.0 = full value)
QS_CESSION = 0.0           # quota share: share of every loss ceded to the reinsurer (0 = no quota share)
XL_RETENTION_KES = 50e6    # cat excess of loss: cedant keeps the first KES 50 m of each flood ...
XL_LIMIT_KES = 250e6       # ... reinsurer pays the next KES 250 m ("250 m xs 50 m")

DEFAULT_TERMS = dict(ded_pct=DED_PCT, ded_min=DED_MIN_KES, limit_pct=LIMIT_PCT, qs=QS_CESSION,
                     retention=XL_RETENTION_KES, limit=XL_LIMIT_KES)
LAYERS = ["ground_up", "policyholder", "gross", "qs_ceded", "xl_ceded", "reinsurer", "net"]
# names follow the hackathon brief's terms: ground-up, deductible, limit, gross, quota share, cat XL, net
LABELS = {"ground_up": "Ground-up loss (total damage)", "policyholder": "Kept by building owners (deductibles)",
          "gross": "Gross loss (after deductible and limit)", "qs_ceded": "Quota share (reinsurer)",
          "xl_ceded": "Cat excess of loss (reinsurer)", "reinsurer": "Paid by reinsurers", "net": "Net loss (insurer keeps)"}


def policy_terms(tiv, ded_pct=DED_PCT, ded_min=DED_MIN_KES, limit_pct=LIMIT_PCT):
    tiv = np.asarray(tiv, float)
    return np.minimum(np.maximum(ded_pct * tiv, ded_min), tiv), limit_pct * tiv


def insured_loss(gu, ded, lim):
    """Per-building insured loss. gu: (n,) or (n, events); ded, lim: (n,)."""
    gu = np.asarray(gu, float)
    ded, lim = (np.asarray(x, float).reshape(-1, *([1] * (gu.ndim - 1))) for x in (ded, lim))
    return np.clip(gu - ded, 0.0, lim)


def treaty(gross, qs=QS_CESSION, retention=XL_RETENTION_KES, limit=XL_LIMIT_KES):
    """Apply the treaty to portfolio gross loss per event (any shape)."""
    gross = np.asarray(gross, float)
    qs_ceded = qs * gross
    retained = gross - qs_ceded
    xl = np.clip(retained - retention, 0.0, limit)
    return dict(qs_ceded=qs_ceded, xl_ceded=xl, reinsurer=qs_ceded + xl, net=retained - xl)


def layer_losses(gu_building, tiv, terms=None):
    """Portfolio loss per event for every layer. gu_building: (n_buildings, n_events) -> dict layer -> (n_events,)."""
    t = {**DEFAULT_TERMS, **(terms or {})}
    ded, lim = policy_terms(tiv, t["ded_pct"], t["ded_min"], t["limit_pct"])
    gross_b = insured_loss(gu_building, ded, lim)
    gu, gross = gu_building.sum(0), gross_b.sum(0)
    return dict(ground_up=gu, policyholder=gu - gross, gross=gross,
                **treaty(gross, t["qs"], t["retention"], t["limit"]))


def ep_table(rps, layers):
    """One row per return period, one column per layer (KES)."""
    return pd.DataFrame({"return_period": rps, **{k: layers[k] for k in LAYERS}})


def _rp_where(rps, loss, level):
    """Return period at which a monotone loss curve reaches `level` (log-interpolated). None if never reached;
    0 if already reached at the most frequent modelled event."""
    loss = np.asarray(loss, float)
    if loss[-1] < level:
        return None
    if loss[0] >= level:
        return 0.0
    i = int(np.argmax(loss >= level))
    f = (level - loss[i - 1]) / (loss[i] - loss[i - 1])
    return float(np.exp(np.log(rps[i - 1]) + f * (np.log(rps[i]) - np.log(rps[i - 1]))))


def layer_metrics(rps, layers, terms=None):
    """What a reinsurance underwriter reads off the layer: expected loss, technical rate on line, when it starts
    paying and when it is used up."""
    t = {**DEFAULT_TERMS, **(terms or {})}
    retained = layers["gross"] - layers["qs_ceded"]
    xl_aal = float(cm.aal_from_ep(rps, layers["xl_ceded"]))
    return dict(xl_aal=xl_aal, rate_on_line=xl_aal / t["limit"] if t["limit"] > 0 else float("nan"),
                attach_rp=_rp_where(rps, retained, t["retention"]),
                exhaust_rp=_rp_where(rps, retained, t["retention"] + t["limit"]),
                aal={k: float(cm.aal_from_ep(rps, layers[k])) for k in LAYERS})


def simulate(d, terms=None, n_sims=500, vary_depth_scale=True, seed=42, tier_rp=None):
    """Monte Carlo ranges for every layer, using the same draws as catmodel.simulate (same seed)."""
    rps = cm.hazard_scores(d, tier_rp)[1]
    out = {k: np.empty((n_sims, len(rps))) for k in LAYERS}
    tiv = d.tiv_kes.to_numpy()
    for s, gu_b in enumerate(cm.simulated_building_losses(d, n_sims, vary_depth_scale, seed, tier_rp)):
        for k, v in layer_losses(gu_b, tiv, terms).items():
            out[k][s] = v
    return rps, out


def who_pays(layers, j):
    """Split of one event's ground-up loss between policyholders, cedant and reinsurer (sums to ground-up)."""
    return {"Building owners (deductibles)": float(layers["policyholder"][j]),
            "Insurer (net loss)": float(layers["net"][j]),
            "Reinsurer: quota share": float(layers["qs_ceded"][j]),
            "Reinsurer: cat excess of loss": float(layers["xl_ceded"][j])}
