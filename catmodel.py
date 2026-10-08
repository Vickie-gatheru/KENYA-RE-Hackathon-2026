"""Nairobi pluvial flood CAT model - core engine (v2, with uncertainty).

Pipeline:  exposure + proxy hazard  ->  hazard (score -> depth)  ->  vulnerability (depth -> damage ratio)
           ->  financial engine (damage x value, summed per scenario)  ->  EP curve + AAL
           ->  Monte Carlo: same chain run N times with damage and depth drawn from ranges.

Labels used throughout:  REAL = from a cited real dataset; PROXY = constructed hazard proxy;
ASSUMED = our modelling choice; SYNTHETIC = generated exposure.
Structure follows the Oasis LMF four-stage approach (hazard / vulnerability / exposure / financial).
"""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm

# ---------------------------------------------------------------- ASSUMPTIONS (edit here)
TIERS = ["common", "occasional", "moderate", "severe", "extreme"]
# ASSUMED: tier -> return period. Widest map ('common') = rarest event (per dataset metadata);
# values follow the organisers' reference dashboard.
TIER_RP = {"common": 250, "occasional": 100, "moderate": 50, "severe": 25, "extreme": 10}

# ONLY hazard_score_common IS USED. The organisers built the other four tiers by keeping the top 30/20/10/5% of
# cells of the same underlying score and rescaling to 0-1. We verified on the full raster that each tier equals
#     clip((common - cutoff) / (1 - cutoff), 0, 1)
# exactly (max error < 1e-5), with the cutoffs below (= the common-score value at that share of all map cells).
# So the model derives every event footprint from the common score alone and ignores the other columns.
TIER_CUTOFF = {"common": 0.0, "occasional": 0.0862, "moderate": 0.1729, "severe": 0.2805, "extreme": 0.3742}


def tier_scores(common):
    """Scores for every tier derived from the common score alone. common: array (n,) -> dict tier -> (n,)."""
    c = np.clip(np.asarray(common, float), 0, 1)
    return {t: np.clip((c - k) / (1 - k), 0, 1) for t, k in TIER_CUTOFF.items()}

# ASSUMED: susceptibility score -> depth. depth_m = score x DEPTH_SCALE_M (brief's example: 1.0 -> 4 m).
DEPTH_SCALE_M = 4.0
# ASSUMED: plausible range for that scale, used in the Monte Carlo (triangular min / mode / max).
DEPTH_SCALE_RANGE = (3.0, 4.0, 5.0)

# SOURCE: Huizinga, J., de Moel, H. & Szewczyk, W. (2017) Global flood depth-damage functions: methodology and
# the database with guidelines. EU Joint Research Centre, JRC105688. Africa, residential buildings
# (mean damage factor and standard deviation by water depth). Values as tabulated in the JRC Excel database.
JRC_DEPTH_M = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
JRC_AFRICA_RES_MEAN = np.array([0.0, 0.219925, 0.378227, 0.530589, 0.635637, 0.816940, 0.903435, 0.957152, 1.0])
JRC_AFRICA_RES_STD = np.array([0.0, 0.042004, 0.114296, 0.198396, 0.207822, 0.205247, 0.141856, 0.076209, 0.0])

# ASSUMED (our adaptation, not from JRC): JRC gives one Africa residential curve. We adapt it per building type:
#   damage(depth) = cap x JRC(depth x depth_factor)
# depth_factor > 1 means the building behaves as if the water were deeper (fragile: no plinth, earth floor,
# light walls); < 1 means more resistant (raised plinth, reinforced frame). cap keeps damage below 100% of value
# because land and foundations survive (problem brief: 80-95%). permanent_masonry uses the JRC curve unshifted.
VULN = {
    "informal_iron_sheet": dict(depth_factor=1.6, cap=0.95),
    "semi_permanent":      dict(depth_factor=1.3, cap=0.90),
    "permanent_masonry":   dict(depth_factor=1.0, cap=0.85),
    "concrete_rcc":        dict(depth_factor=0.75, cap=0.80),
}

# SECONDARY (damage) UNCERTAINTY: damage for a building is drawn from a Beta distribution whose mean is the curve
# value and whose standard deviation is the JRC Africa residential standard deviation at that (adjusted) depth.
# ASSUMED: within one flood, buildings' damage "luck" is partly shared (same storm, same drains).
EVENT_CORRELATION = 0.2


# ---------------------------------------------------------------- 1. EXPOSURE
def load_exposure(path):
    """SYNTHETIC exposure with PROXY hazard scores attached."""
    d = pd.read_csv(path)
    # DATA FIX: file tiv_kes is ~10x floor_area x cost; the metadata defines tiv as area x cost rounded
    # to 5,000 (total KES 6,363,470,000). Recompute to match the documentation.
    d["tiv_kes_file"] = d["tiv_kes"]
    d["tiv_kes"] = (d.floor_area_m2 * d.cost_per_m2_kes / 5000).round() * 5000
    return d


def hazard_scores(d, tier_rp=None):
    """(n_buildings, n_events) scores, columns ordered by ascending return period.
    Built ONLY from d['hazard_score_common'] (other hazard_score_* columns are ignored)."""
    tier_rp = tier_rp or TIER_RP
    tiers = sorted(TIERS, key=lambda t: tier_rp[t])
    rps = np.array([tier_rp[t] for t in tiers])
    derived = tier_scores(d["hazard_score_common"].to_numpy())
    return np.column_stack([derived[t] for t in tiers]), rps


# ---------------------------------------------------------------- 2. VULNERABILITY
def class_params(classes):
    p = pd.DataFrame([VULN[c] for c in classes])
    return p.depth_factor.to_numpy(), p.cap.to_numpy()


def damage_ratio(depth, depth_factor, cap):
    """Mean damage ratio: cap x JRC Africa residential curve at (depth x depth_factor). 0 when dry."""
    return cap * np.interp(np.asarray(depth) * depth_factor, JRC_DEPTH_M, JRC_AFRICA_RES_MEAN)


def damage_sd(depth, depth_factor):
    """Standard deviation of the damage factor (JRC scale, before the cap) at the adjusted depth."""
    return np.interp(np.asarray(depth) * depth_factor, JRC_DEPTH_M, JRC_AFRICA_RES_STD)


# ---------------------------------------------------------------- 3+4. FINANCIAL ENGINE
def deterministic(d, depth_scale=DEPTH_SCALE_M, tier_rp=None):
    """Central estimate: one loss per building per return period."""
    score, rps = hazard_scores(d, tier_rp)
    fac, cap = (a[:, None] for a in class_params(d.housing_class))
    dr = damage_ratio(score * depth_scale, fac, cap)
    loss = dr * d.tiv_kes.to_numpy()[:, None]
    return dict(rps=rps, depth=score * depth_scale, dr=dr, loss=loss)


def aal_from_ep(rps, port_loss):
    """Average annual loss from losses at given return periods (trapezoid over exceedance probability).
    Adds the tail beyond the rarest RP at that RP's loss; events more frequent than the most frequent
    RP are taken as zero loss, so this is a LOWER estimate."""
    p = 1.0 / rps                               # descending as rps ascend
    order = np.argsort(p)
    lp = np.take(port_loss, order, axis=-1)
    area = np.trapezoid(lp, p[order], axis=-1)
    return area + lp[..., 0] * p[order][0]


def simulate(d, n_sims=2000, vary_depth_scale=True, seed=42, tier_rp=None):
    """Monte Carlo over damage uncertainty (and optionally the score->depth scale).
    Within a simulation a building keeps the same random quantile across return periods,
    so loss always rises with rarity. Returns (rps, portfolio loss per simulation x return period)."""
    rps = hazard_scores(d, tier_rp)[1]
    out = np.empty((n_sims, len(rps)))
    for s, loss in enumerate(simulated_building_losses(d, n_sims, vary_depth_scale, seed, tier_rp)):
        out[s] = loss.sum(axis=0)
    return rps, out


def simulated_building_losses(d, n_sims=2000, vary_depth_scale=True, seed=42, tier_rp=None):
    """Yields one (n_buildings, n_events) ground-up loss matrix per simulation. Shared by simulate() and the
    insurance/reinsurance engine (financial.py), so both see identical draws for the same seed."""
    rng = np.random.default_rng(seed)
    score, rps = hazard_scores(d, tier_rp)
    fac, cap = (a[:, None] for a in class_params(d.housing_class))
    tiv = d.tiv_kes.to_numpy()[:, None]
    n = len(d)
    for _ in range(n_sims):
        scale = rng.triangular(*DEPTH_SCALE_RANGE) if vary_depth_scale else DEPTH_SCALE_M
        depth = score * scale
        mean_dr = damage_ratio(depth, fac, cap)                         # (n, rps)
        # correlated uniform per building: shared event shock + own shock
        z = np.sqrt(EVENT_CORRELATION) * rng.standard_normal() + \
            np.sqrt(1 - EVENT_CORRELATION) * rng.standard_normal(n)
        u = norm.cdf(z)[:, None]
        m = np.clip(mean_dr / cap, 1e-6, 1 - 1e-6)                    # mean on the JRC [0, 1] scale
        v = np.clip(damage_sd(depth, fac) ** 2, 1e-6, 0.95 * m * (1 - m))  # JRC variance, kept Beta-feasible
        common = m * (1 - m) / v - 1
        a, b = m * common, (1 - m) * common
        dr = np.where(mean_dr > 0, cap * beta_dist.ppf(u, a, b), 0.0)
        yield dr * tiv


def summarise(rps, sims):
    q = np.percentile(sims, [5, 50, 95], axis=0)
    aal = aal_from_ep(rps, sims)
    return dict(
        mean=dict(zip(rps.tolist(), sims.mean(0).tolist())),
        p5=dict(zip(rps.tolist(), q[0].tolist())),
        p50=dict(zip(rps.tolist(), q[1].tolist())),
        p95=dict(zip(rps.tolist(), q[2].tolist())),
        aal_mean=float(aal.mean()), aal_p5=float(np.percentile(aal, 5)), aal_p95=float(np.percentile(aal, 95)),
    )


def loss_at_rp(rps, port_loss, target):
    """Portfolio loss at a target return period, interpolated linearly in log(return period).
    Returns NaN outside the modelled range (no extrapolation)."""
    rps = np.asarray(rps, float)
    if target < rps.min() or target > rps.max():
        return float("nan")
    return float(np.interp(np.log(target), np.log(rps), port_loss))


# ASSUMED alternatives for the tier -> return period mapping, used only for sensitivity testing.
RP_MAPPINGS = {
    "reference (10-250y)":  {"extreme": 10, "severe": 25, "moderate": 50, "occasional": 100, "common": 250},
    "more frequent (5-100y)": {"extreme": 5, "severe": 10, "moderate": 25, "occasional": 50, "common": 100},
    "rarer (25-500y)":      {"extreme": 25, "severe": 50, "moderate": 100, "occasional": 250, "common": 500},
}
