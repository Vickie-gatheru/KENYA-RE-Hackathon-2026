# Nairobi Urban Flood CAT Model: Architecture

The model follows the Oasis LMF four-stage structure (hazard → vulnerability → exposure → financial).
It is a hackathon prototype built on open data, a constructed hazard proxy and a synthetic portfolio.
Losses are computed **ground-up**, then passed through ASSUMED policy terms (insured / cedant gross) and an ASSUMED
reinsurance treaty (ceded / net) by `financial.py` - see "Financial engine" below.

```
   [Copernicus GLO-30 DEM + OSM rivers]  PROXY            [Cited flood reports]  REAL TEXT
                    │                                              │
                    ▼                                              ▼
   ┌──────────────────────────────────┐        ┌──────────────────────────────────────────┐
   │ 1. HAZARD (base)                 │        │ 2. AI DRAINAGE LAYER                     │
   │  5 susceptibility tiers, 0-1     │        │  a. LLM extracts place, mechanism,       │
   │  tier → return period (ASSUMED)  │        │     severity, verbatim evidence quote    │
   │  extreme 10y … common 250y       │        │  b. quotes checked against source text;  │
   └────────────────┬─────────────────┘        │     unverifiable signals rejected        │
                    │                          │  c. places geocoded via OSM Nominatim    │
                    │                          │     (never from the validation file)     │
                    │                          │  d. uplift ΔS = w·exp(−d²/2σ²),          │
                    │                          │     max over sites, score capped at 1    │
                    │                          └───────────────────┬──────────────────────┘
                    └─────────────────► effective hazard S_eff ◄───┘
                                               │
   [600 SYNTHETIC buildings] ──► 3. EXPOSURE: class, insured value (area × cost/m²)
                                               │
                                               ▼
                         4. VULNERABILITY: depth = S_eff × 4 m (ASSUMED)
                            JRC Africa residential depth-damage curve (published), adapted per type
                                               │
                                               ▼
                         5. FINANCIAL ENGINE: loss = insured value × damage ratio
                            EP curve, AAL, 2,000-run Monte Carlo (damage + depth-scale uncertainty)
                                               │
                         6. VALIDATION: hotspot recall on 24 county hotspots (held out)
                            uniform-weight ablation, uplift footprint, sensitivity grids
                                               │
                                               ▼
                         7. DASHBOARD: EP chart, maps, class breakdown, AI evidence trail
```

## Main use: evaluating an incoming risk or flood claim (`evaluate.py`, first dashboard tab)

An underwriter enters a location (place name, coordinates or an example), building type, insured value and, for a
claim, the claimed loss. The evaluation compares the site with the hazard map **and with nearby assessed assets** in
`exposure_nairobi_with_hazard.csv`, and explains every step:

1. **Map at the site** - `hazard_score_common` at the exact 31 m cell, plus the 500 m maximum and 1 km average.
2. **Nearby assessed assets** - up to 10 nearest within 1 km (expands if fewer than 3), weighted by distance
   (Gaussian, σ = radius/2); their mapped `hazard_score_common` values are averaged.
3. **Blend** - site weight 60% for exact coordinates, 35% for a place name (which geocodes to an area centre).
4. **AI adjustment** (optional) - drainage-evidence / ML uplift at the site.
5. **Events** - the blended score → event footprints (1-in-10 … 1-in-250) → depth → JRC damage for the type → loss.
6. **Price and impact** - technical premium and rate vs portfolio; map-only and neighbours-only premiums for
   comparison; change in portfolio 1-in-100 loss; insured value already within 1 km.
7. **Claim check** - claimed loss as % of value vs modelled damage: *Consistent* (with implied return period),
   *Unusually severe*, *Inconsistent* (above the type's damage cap) or *Not supported by the map* (with the
   caveat that the map cannot see drainage flooding).

Output: risk band (by city percentile), confidence (location precision, number of neighbours, map-neighbour
agreement), flags, a map, the score breakdown, tables of neighbours and events, a downloadable report, and an
optional AI-written underwriting note with a number check.

## Only `hazard_score_common` is used

The organisers built the other four tiers by keeping the top 30/20/10/5% of cells of the same score and rescaling.
We verified on the full raster that each equals `clip((common − cutoff)/(1 − cutoff), 0, 1)` exactly, with cut-offs
0.0862 / 0.1729 / 0.2805 / 0.3742. The engine therefore reads only `hazard_score_common` and derives each event's
footprint itself; a test confirms that scrambling the other columns changes nothing.

## Who it is for: underwriters

The dashboard (`app.py`) answers five underwriting questions:

| Question | Where |
|---|---|
| How bad could it get? | Header: 1-in-100 and 1-in-250 loss (PML) with ranges; EP curve |
| What should I charge? | Pricing: technical premium (AAL) and rate per mille, by building type and per building |
| Where am I concentrated? | Accumulation: map + 2 km zones ranked by share of the 1-in-100 loss |
| Should I accept this risk? | Quote a new risk: broker text → LLM → structured rows → hazard read from the maps → premium, rate vs portfolio, change in portfolio PML/AAL, concentration within 1 km, flags |
| Explain it to me | Underwriter memo: LLM writes from a fixed list of model facts; any number not in the facts is flagged |

Technical premium = modelled average annual loss, ground-up: a pricing floor before expenses and profit.

## Why the AI layer

The starter proxy sees terrain and river proximity, so it flags 12 of the 24 county hotspots, mostly the
Eastlands river valleys. The other 12 (Kibera, Westlands, Lavington and others) flood because of
drainage-system issues the proxy cannot detect. The LLM reads real reports and extracts structured,
quote-backed evidence of where drainage fails. A fixed, documented formula turns that evidence into a
hazard change, and the whole loss chain re-runs. The LLM supplies evidence, not numbers.

## Underwriting assistant (`agent.py`, "Ask the assistant" tab)

A chat assistant for underwriters that combines **RAG** and **tool use**, and shows its proof:
- **Tools = the live model**: portfolio summary and EP curve, breakdown by building type or zone, price a new
  risk (by place name or coordinates), explain a location (proxy scores, ML/SHAP reasons, nearby flood reports
  with quotes), what-if on assumptions, hotspot check.
- **RAG = the model's own knowledge**: TF-IDF retrieval over `ARCHITECTURE.md`, `docs/*.md` (underwriter guide)
  and every extracted flood report quote. Add more `.md`/`.txt` files to `docs/` to extend it.
- **Proof**: every answer has a panel listing each tool call with its full result and each retrieved passage.
  Numbers in the answer that don't match a tool result (to within rounding) are flagged.
- Works with any provider in `llm.py` via a simple JSON tool protocol. `LLM_PROVIDER=test` is an offline scripted
  stub used only to test the interface.

## Machine learning: flood-susceptibility model (`ml_hazard.py`)

The evidence layer only raises hazard where a report names a place. The ML model generalises:

- **Labels (weak supervision):** places the LLM extracted from reports = positives; random city locations
  >1 km from any positive = background (positive-unlabelled learning: "not reported", not "never floods").
- **Features (`features.py`):** proxy score at the spot, 1 km neighbourhood mean, 500 m maximum; plus, from
  OpenStreetMap (`fetch_osm.py`): distance to rivers, distance to mapped drains, road density (built-up/paved
  proxy), informal-settlement areas.
- **Models:** logistic regression and gradient boosting; the better on **spatial cross-validation** (3 km blocks)
  is used.
- **Held-out test:** the 24 county hotspots are never used in training. We report whether they score above
  ordinary city locations (AUC) and how many fall in the city's top 10% / 20%, against the proxy alone.
- **Baselines for the held-out test:** each feature on its own (if one plain feature matches the model, the model
  adds little); the same test against only the built-up half of the city (road density ≥ city median), since
  hotspots are all in built-up Nairobi and the city sample includes parkland; and a **buffer test** that retrains
  without any training place within 1 / 2 km of a hotspot (news and the county list name the same neighbourhoods).
- **Use in the model:** locations in the city's top 10% by ML score get a hazard uplift up to w_ML = 0.30,
  scaled by (1 − existing score) so the proxy's wet areas aren't double-counted. With evidence sites also on,
  the larger uplift wins.
- **Explanation:** SHAP values per location ("1.2 km from the nearest mapped drain - raises risk"), shown for
  any hotspot/building and for every quoted risk; global importance across the city.

## How we keep the validation honest

| Risk | Control |
|---|---|
| Grading on the data we tuned on | Uplift placed only from report text. The 24 hotspot coordinates are used only for scoring. |
| LLM reads the answer key | Sources naming ≥10 of the 24 hotspots are auto-excluded (the county list itself). |
| LLM invents evidence | Each signal must quote its source verbatim. Paraphrases and invented places are rejected and logged. |
| LLM mistakes a forecast for evidence | A quote that looks ahead (alert, forecast, warning) and reports no flooding that happened is rejected (`extract.is_forecast_only`); the prompt also forbids forecasts. |
| Gaming recall by uplifting everywhere | Uplift footprint (% of buildings and value affected) is reported next to recall. |
| LLM adds nothing | Ablation: same sites with uniform weight. If results match, the grading added nothing. |
| Double-counting river flooding | `river_overflow` signals are excluded, since the proxy already captures them. |
| Any 41 sites would do as well | Placebo (`ai.placebo_recall`): same sites and weights moved to random portfolio buildings / city points, 500 times; p = share of random runs scoring at least as well. Buildings are the fair pool: reports and hotspots both cluster where people live. |
| ML only learns "built-up vs empty" | Single-feature AUCs and a built-up-only test sit next to the headline AUC. |
| Training places sit next to test hotspots | ML buffer test: retrain without training places within 1 / 2 km of any hotspot. |

### Results of the first real run (7 Oct 2026; 16 sources fetched, `openai/gpt-oss-120b` on Groq)

- 84 signals kept, 21 rejected (8 quote not verbatim, 1 place name, **12 forecast or warning**); 76 geocoded; 36
  evidence sites (river_overflow excluded). 39 signals have mechanism "unknown" and are uplifted (judgement call).
- **Forecast filter (added after review):** the first pass kept 10 signals from one sentence - "Weather alerts had
  indicated widespread heavy rain across ... Mathare" - which the LLM had labelled drainage blockage. It was copied
  word for word, so the verbatim check passed, but it is a forecast, not a flood report. `extract.validate` now rejects
  quotes that look ahead (alert, forecast, warning, expected to, ...) and report no flooding that happened;
  "flood-prone" statements are kept. Re-run from the cached LLM replies (no new calls).
- The 24-hotspot list article (`star_dam_evacuation`) was auto-skipped. The police warning article
  (`kenyans_nps_warning`, 6 of 24 hotspots) contributed only river_overflow signals, so it has no effect on uplift.
- **Hotspot recall:** proxy 12/24 → evidence layer 19/24 (unchanged by the filter: the forecast signals detected
  nothing); uniform-weight ablation 20/24 (**the LLM's severity and confidence weighting adds nothing to recall** - the
  value is in *which* places it finds). Footprint: 6.7% of the city map, 22% of buildings (was 7.5% / 25%).
- **Placebo:** random city points average 13.1/24 (p < 0.002); random portfolio buildings average 15.2/24,
  **p = 0.03** (was 0.048 with the forecast signals) - better than chance against the fair baseline, by a modest margin.
- **ML:** logistic regression, 51 places, spatial-CV AUC 0.95; held-out hotspot AUC 0.91 vs proxy 0.54. But
  **road density alone scores 0.89**. On the built-up half of the city: ML 0.83, road density 0.79, proxy 0.53 - a
  modest real gain. Buffer test: AUC 0.92 with training places within 2 km of a hotspot removed (18 left),
  so the result is not driven by news naming the same places. The OSM informal-settlement layer has only 14
  polygons for Nairobi; the feature is almost always 0 (AUC alone 0.50, zero SHAP weight) and changes nothing.
- Portfolio with evidence uplift: 1-in-100 KES 327 m → 425 m; AAL KES 13.8 m → 19.3 m.

## Financial engine: insured and reinsured loss (`financial.py`, "Insurance & reinsurance" page)

Per building: insured = min(max(ground-up − deductible, 0), limit). Per flood event, on the cedant's gross portfolio
loss G: quota share ceded = q·G; cat XL ceded = min(max((1−q)·G − retention, 0), limit); net = (1−q)·G − XL.
Each event's financial loss is a monotone function of its ground-up loss, so applying terms to each return-period
scenario gives the financial EP curves directly; the Monte Carlo reuses the ground-up engine's draws
(`catmodel.simulated_building_losses`), so all ranges are consistent. Layer outputs: expected loss, technical rate on
line (expected loss ÷ limit, before the reinsurer's loads), and the return periods at which the layer attaches and
is exhausted. A "who pays" chart splits one flood between policyholders, cedant and reinsurer.

**All terms are ASSUMED** - the exposure file has none. Defaults (round numbers scaled to the synthetic book):
deductible 2% of value (min KES 10,000), limit = full value, no quota share, cat XL KES 250 m xs 50 m per flood.
Simplifications: 100% take-up, one flood per year (occurrence = aggregate), no reinstatements, no surplus treaty.
Proxy-only results with these defaults: insured AAL KES 12.2 m (ground-up 13.8 m); layer attaches ≈1-in-12,
exhausted ≈1-in-107; layer expected loss KES 6.4 m/yr, technical rate on line 2.6%.

## Assumptions (all editable in `catmodel.py` / `hazard_ai.py` / `financial.py`)

| Assumption | Value | Basis | Tested by |
|---|---|---|---|
| Tier → return period | extreme 10, severe 25, moderate 50, occasional 100, common 250 y | Organisers' reference dashboard; widest map = rarest event (metadata) | 3 mappings, weighted 50/25/25 inside the Monte Carlo |
| Score → depth | depth = score × 4 m | Brief's example | Monte Carlo, 3–5 m |
| Base vulnerability | JRC Africa residential mean damage by depth (0–6 m) | PUBLISHED: Huizinga et al. 2017, JRC105688 | — |
| Per-type adjustment | damage = cap × JRC(depth × factor): informal ×1.6/95%, semi-permanent ×1.3/90%, masonry ×1.0/85%, RCC ×0.75/80% | ASSUMED: fragile types behave as if water were deeper; caps per brief (80–95%) | — |
| Damage spread | Beta with JRC Africa residential standard deviation at each depth | PUBLISHED (same JRC table) | Monte Carlo |
| Event correlation | 0.2 | Judgement | — |
| Uplift strength / reach | w_max = 0.30, σ = 0.75 km | Neighbourhood scale (hotspot coords are area centres) | 3 × 3 grid |
| Detection threshold | τ = 0.05 | Judgement | — |
| Insured value | area × cost/m², rounded to 5,000 | Metadata definition (file column was ~10× this) | — |
| Policy terms | deductible 2% of value (min KES 10,000); limit = value; 100% take-up | ASSUMED (no terms in the data) | live controls |
| Reinsurance | cat XL KES 250 m xs 50 m per flood; quota share 0% | ASSUMED, illustrative | live controls |

## Uncertainty (taxonomy from the Oasis LMF introduction, p. 8)

- **Model:** vulnerability curves, depth scale, tier → return period mapping. The mapping is the largest
  single driver: AAL is KES 6.1 m, 13.8 m or 29.3 m under the three mappings tested.
- **In the ranges (`catmodel.with_rp_uncertainty`):** the mapping does not change any tier's loss, only how often
  it happens, so each Monte Carlo run also draws one mapping (ASSUMED weights: reference 50%, more frequent 25%,
  rarer 25%) and keeps its damage and depth draws. Each run's AAL uses its own mapping; its loss curve is re-read at
  the reference return periods (log-interpolated; 0 below the mapping's most frequent event, held at its rarest beyond
  it - lower bounds, so the top of the 1-in-250 range is understated). The central estimate stays on the reference
  mapping. Map only, 2,000 runs: **AAL range KES 5.2-33.3 m** (damage + depth alone: 10.2-17.6 m), mean 15.5 m vs 13.8 m
  central; **1-in-100 range KES 170-570 m** (alone: 249-408 m). Used in the dashboard, `summary.json`
  (`baseline.incl_return_period_uncertainty`), the saved EP chart and the assistant.
- **Data:** synthetic exposure; approximate hotspot and place coordinates (area centres).
- **Unmodelled:** contents, business interruption, rainfall intensity, drainage network geometry,
  flooding not reported in our text sources.

## Files

| File | Role |
|---|---|
| `catmodel.py` | Engine: exposure, hazard, vulnerability, ground-up loss, Monte Carlo, AAL |
| `financial.py` | Insured (deductible/limit) and reinsured (quota share + cat XL) losses, layer metrics |
| `public_app.py` | Public estimate page for building owners and cedants (own port): one building's indicative flood cost, no portfolio data shown, no LLM |
| `briefing.py` | Structured evaluation briefing (headline, drivers, actions, questions, caveat): LLM-written with every number checked, or rule-based; stance always by rules |
| `pin_picker.py` | Search a place or click the map to drop a pin (pydeck click selection on an invisible grid) |
| `submission.py` | Reads a broker's submission PDF: each fact found as a verbatim quote (rules, then LLM for gaps; values parsed from the quote by code), rule-based consistency checks (address vs GPS, landmarks, river height, floor areas, services, basement plant, deadline), ASSUMED building-shape losses (basements flood at 5 cm street water, ~3 m deep; upper floors dry), broker-says vs model-says |
| `review_agent.py` | Submission review agent: LLM (or a rule-based plan) chooses deterministic tools - price the other location, basement sensitivity, flood-map-only price, nearby flood reports, broker questions - then writes a summary checked against tool results; drafts (never sends) a broker email |
| `importer.py` | Loads any insurer's building list (CSV/Excel): guesses columns, maps free-text building types for review, uses a flood-score column if present (else reads the flood map), applies the starter kit's floor area x cost value fix, rejects unusable rows with a reason each |
| `ml_select.py` | ML model selection on training data only: spatial CV against built-up background points; the 24 hotspots are scored for transparency, never used to choose. `quick=True` is the smaller grid the workspace uses |
| `scraper.py` | Finds recent flood news for a city (Bing News RSS; GDELT when it answers), drops repeats, reads each article, LLM extraction with extract.py's checks (verbatim quote, no forecasts), merges places across articles, geocodes - output is candidates for review |
| `workspace.py` | Model workspace back-end: one folder per region (regions/, git-ignored) - flood map, portfolio, OSM layers, reviewed flood reports, model versions; train / activate / results. Points hazard.py and features.py at a region and back (Nairobi defaults unchanged) |
| `workspace_app.py` | The workspace in the browser (`streamlit run workspace_app.py --server.port 8503`): flood map, portfolio, map layers, flood reports (search + approve), build model, results, history - no terminal |
| `decisions.py` | Underwriting decisions on the Evaluate page (Accept / Accept with loading / Refer / Decline; claims: Pay / adjuster / query / decline), logged beside the model's suggestion; accepted risks are written into the book (Flood briefing, Accumulation, Reinsurance include them); a PAID claim becomes approved flood evidence in the region, removed again if the decision changes |
| `quote_requests.py` | Public estimate page -> 'Ask for a quote' -> the underwriters' queue on Evaluate; one click evaluates the request, the recorded decision closes it (kept on this computer only) |
| `brand.py`, `.streamlit/config.toml` | Kenya Re website palette and fonts (colours only, no logo); validated chart palette |
| `extract.py` | LLM extraction + validation (provider: groq / gemini / anthropic / openai / manual paste) |
| `geocode.py` | Place → coordinates via Nominatim (run on a laptop with internet) |
| `hazard_ai.py` | Uplift, hotspot recall, footprint, ablation |
| `run.py` | Batch run; writes `out/summary.json`, CSVs, `ep_curve.png` |
| `app.py` | Underwriting workbench (Streamlit); assumptions are live sliders; offline map mode for the venue |
| `hazard.py` | Reads the five proxy hazard maps and samples them at any coordinates |
| `underwriting.py` | Technical pricing, accumulation zones, risk quoting, free-text parsing, memo number check |
| `features.py` | ML features at any coordinate (hazard-map neighbourhood stats + OpenStreetMap) |
| `ml_hazard.py` | ML flood-susceptibility model: training, spatial CV, held-out hotspot test, uplift, SHAP |
| `fetch_osm.py` | Downloads OpenStreetMap rivers, drains, roads, informal areas (laptop) |
| `evaluate.py` / `evaluate_page.py` | Risk & claim evaluation against the map and nearby assets, with step-by-step explanation (main page) |
| `agent.py` | Underwriting assistant: model tools + RAG + proof trace + number check |
| `docs/underwriter_guide.md` | Plain-English guide used by the assistant's retrieval |
| `llm.py` | One LLM interface: groq / gemini / anthropic / openai |
| `fetch_sources.py` | Downloads the reports in `data/sources.csv` (excludes the 37-hotspot list articles) |
| `tests/test_pipeline.py` | 56 invariant checks (engine, common-only derivation, evaluation, AI layer, rasters, underwriting, ML, assistant) on an invented, labelled TEST fixture |

## Run

Starter files (exposure, hotspots, and the five hazard maps in `data/hazard/`) are already included. `data/hazard/hazard_grids.npz` is a lossless numpy copy of the maps, used automatically if rasterio/GDAL will not load (a common Windows DLL problem).

Fastest route to real AI results: `python run_ai.py` (runs every AI step in order and prints the before/after evidence plus a review checklist).
For the LLM features in the dashboard, set the provider in the same terminal before `streamlit run`.

```
pip install -r requirements.txt

# baseline dashboard - works immediately
streamlit run app.py

# AI layer (laptop with internet)
python fetch_sources.py                    # downloads reports listed in data/sources.csv -> sources/
                                           #   paywalled/short pages are listed: paste their text by hand
$env:LLM_PROVIDER="groq"                  # PowerShell; or gemini / anthropic / openai / manual
$env:GROQ_API_KEY="gsk_..."                # free key from console.groq.com/keys
$env:LLM_MODEL="openai/gpt-oss-120b"     # optional; any large Groq chat model with JSON mode
python extract.py                          # -> out/signals_raw.json (responses cached in out/responses/)
python geocode.py                          # -> data/signals.csv ; check data/gazetteer.csv for MISSING rows
python fetch_osm.py                        # optional but recommended: drains, rivers, roads, informal areas
python ml_hazard.py                        # trains + validates the ML model -> out/ml_model.pkl, out/ml_metrics.json
streamlit run app.py                       # AI controls appear automatically once data/signals.csv exists
python run.py                              # optional: same results as files (out/summary.json etc.)
python tests/test_pipeline.py
```

Sources: Oasis LMF, *Introduction to Catastrophe Modelling*
(https://oasislmf.org/application/files/6917/1624/0852/Intro_toCatModelling.pdf);
hackathon problem statement, step-by-step guide and dataset metadata.
