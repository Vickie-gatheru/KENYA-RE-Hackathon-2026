# Nairobi Flood Risk Workbench

**Kenya Re AI4I Hackathon 2026 · Team A** - a flood catastrophe model for Nairobi with an underwriting workbench on
top: price a risk, review a broker's submission, manage the portfolio and its reinsurance, and load a new dataset or
city - all in the browser, with every AI step checked and a person in control.

> Prototype built for the hackathon - not a Kenya Re product. The portfolio is **SYNTHETIC**, the flood map a
> **PROXY** (relative susceptibility, not measured depth) and policy/treaty terms **ASSUMED**; the app labels each.

## Who it is for

| User | What they get |
|---|---|
| **Underwriter** | *Evaluate*: price a new risk or check a claim - a suggested action, the reasons, a map and the loss in each flood size. Read a broker's PDF submission (facts quoted, red flags, a review agent that drafts questions). Record the decision - accepted risks join the book. |
| **Portfolio manager** | *Flood briefing* and *Asset register*: every asset on a map with its flood risk and expected cost, where the risk is concentrated, *Accumulation* zones, *Reinsurance* (who pays, cat XL cost) and a one-click **El Niño stress test**. |
| **Cedant / data team** | *Model workspace*: upload an insurer's book (any CSV/Excel) or a new city's flood map, search the news for flood reports, approve them, train and validate a model, then run the whole dashboard on it. |
| **Building owner** | A public estimate page: expected flood cost per year for one building, and "ask for a quote", which reaches the underwriters' queue. |

## Results (starter portfolio: 600 synthetic buildings, KES 6.4 bn insured)

| | Flood map only | With flood reports + ML |
|---|---|---|
| County flood hotspots detected (24, held out) | 12 / 24 | 21 / 24 |
| Expected loss per year (AAL) | KES 13.8 m | KES 25.2 m |
| 1-in-100 portfolio loss | KES 327 m | KES 601 m |

- The 24 Nairobi County hotspots are **never** used in training or model selection - they are the held-out test.
  The evidence layer beats random placement (placebo p = 0.03 against random portfolio buildings, p < 0.002 against
  random city points).
- ML flood model: held-out AUC 0.913 (flood map alone 0.536); within built-up areas only, 0.846 against 0.817 for the
  best single feature - chosen by spatial cross-validation on training data alone (`ml_select.py`).
- Ranges: the AAL spans roughly KES 10-60 m once Monte Carlo and the uncertain return periods of the flood tiers are
  included; the app shows these ranges beside every headline figure.

## Quick start (Windows PowerShell)

```powershell
cd C:\nairobi_cats
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# optional - only for reading flood reports, broker descriptions, briefings and the assistant
$env:LLM_PROVIDER = "groq"            # groq | anthropic | openai | gemini
$env:GROQ_API_KEY = "<your key>"      # never commit keys; .env is git-ignored
$env:LLM_MODEL    = "openai/gpt-oss-120b"

streamlit run app.py                                      # the workbench  -> http://localhost:8501
streamlit run public_app.py --server.port 8502            # public estimate page
streamlit run workspace_app.py --server.port 8503         # the model workspace on its own (also inside app.py)
python tests/test_pipeline.py                             # 113 invariant checks
```

Without an LLM everything still runs: AI wording falls back to rule-based text, and the news search re-reads saved
articles. Restart Streamlit after changing code or retraining the model.

## The pages

**Underwrite** - *Flood briefing* (portfolio headline, quick actions) · *Asset register* · *Evaluate* (proposal, claim
or broker submission; record decisions; quote requests from the public page) · *Accumulation* · *Reinsurance* ·
*Assistant* (questions answered from the model, numbers checked, with citations).
**Data & models** - *Model workspace*: flood map → portfolio → map layers → flood reports (search + approve) → build
model (versions, model report) → results → *Use in dashboard*.
**Model analysis** (switch on *Analyst mode*) - AI flood evidence, ML flood model, assumptions as live controls.
Sidebar: **El Niño stress test** (ASSUMED: floods about twice as frequent and 25% deeper) and *Analyst mode*.

## How it works

```
INPUT                      PROCESSING                       AI                              OUTPUT
flood hazard maps   ->  import + checks             ->  LLM flood evidence           ->  price, suggested action,
insurer portfolio       score -> water depth            (exact quotes, approved)          recorded decision
broker PDFs             JRC damage -> loss              ML flood model (spatial CV,      portfolio, asset register,
news, OSM, claims       loss curve, AAL, Monte Carlo     SHAP)                            accumulation, reinsurance
                        deductible, quota share, XL     submission review agent          public estimate, reports
          ^                                             assistant & briefings                        |
          +------- closed loop: written risks join the book; paid claims & approved reports retrain --+
```

1. **Hazard** - the organisers' five-tier proxy maps (built from Copernicus GLO-30 terrain and OpenStreetMap rivers).
   Only `hazard_score_common` is used; each flood size is derived from it with the organisers' tier cut-offs.
2. **AI drainage layer** - the flood map cannot see blocked drains. An LLM reads news and research, keeps a place only
   with a word-for-word quote (forecasts and warnings rejected), places are geocoded, and nearby scores rise. An ML
   model learns what reported places have in common (terrain, drains, rivers, built-up density) and scores the city.
3. **Vulnerability** - depth = score × 4 m (ASSUMED); JRC Africa residential depth-damage curve, adjusted per type.
4. **Loss and finance** - loss per building per flood size; loss curve and AAL; Monte Carlo for damage, depth and
   return-period uncertainty; deductible, quota share and cat excess of loss.

The rule throughout: **the LLM supplies evidence and wording, never numbers** - every number shown comes from the
model, and AI text with a number the model did not produce is dropped. Details: [ARCHITECTURE.md](ARCHITECTURE.md);
for underwriters: [docs/underwriter_guide.md](docs/underwriter_guide.md).

## Project layout

| Area | Files |
|---|---|
| Loss model | `catmodel.py` (hazard → depth → damage → loss, Monte Carlo), `financial.py` (policy and treaty), `hazard.py` |
| AI | `extract.py` (LLM evidence), `hazard_ai.py` (evidence uplift), `ml_hazard.py` + `ml_select.py` + `features.py` (ML model), `agent.py` (assistant), `review_agent.py`, `briefing.py`, `submission.py` (broker PDFs), `llm.py` |
| App | `app.py`, `evaluate_page.py`, `portfolio_page.py`, `workspace_page.py`, `public_app.py`, `brand.py`, `pin_picker.py` |
| Data in / records | `importer.py` (any portfolio file), `workspace.py` + `scraper.py` (regions, news search), `decisions.py`, `quote_requests.py`, `store.py` (SQLite database) |
| Pipelines | `run_ai.py` (fetch → extract → geocode → OSM → train → results), `run.py` (results to `out/summary.json`) |
| Data | `data/` - hazard maps, starter portfolio, county hotspots (validation only), approved flood signals, OSM layers |

**Storage.** Decisions, approvals, model versions, quote requests and an audit trail live in one SQLite file,
`regions/kenyare_flood.db`; uploaded datasets, article text and trained region models sit beside it in `regions/`.
`regions/`, `.env` and `sources/*.txt` (article text) are git-ignored.

## Limitations and next steps

- The flood map is a susceptibility proxy, and the return periods of its tiers are assumed - calibrating against
  real flood depths or claims is the biggest accuracy gain available.
- Within built-up areas the ML model is only modestly better than road density alone; more flood reports (the news
  search and paid claims add them) help more than more tuning.
- One user at a time: the flood map is held in the server's memory, so switching regions affects everyone on that
  server. A multi-user version needs per-session regions and PostgreSQL in place of SQLite.
- Pinned versions matter: saved models are tied to scikit-learn 1.9.1 (`requirements.txt`).

## Team

Dylan Gathia (AI & ML engineering) · Patrick Njoroge (catastrophe modelling & data) · Vickie Gatheru (product &
full-stack) - Computer Science.
