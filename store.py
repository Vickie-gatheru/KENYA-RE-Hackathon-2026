"""One SQLite database for everything the system decides, approves or records - regions/kenyare_flood.db (git-ignored).

Tables
  regions         one row per region: its settings (name, country, area, active model, map / portfolio summary)
  events          the audit trail: what happened, when, in which region (portfolio loaded, reports approved, model
                  trained / made active, decisions recorded, claims paid, quote requests ...)
  candidates      flood places found by the scraper or added by paid claims, with their review status
  models          trained model versions and their validation scores (the model files stay on disk)
  gazetteer       place -> coordinates cache per region
  decisions       underwriting and claim decisions, per book
  quote_requests  requests from the public estimate page
Large files stay files, in each region's folder: flood maps, portfolios, OpenStreetMap layers, article text, models.

SQLite is built into Python (no server, no install) and handles several people writing at once safely (write-ahead
log, busy timeout), which the CSV files it replaces did not. The same tables map one-to-one onto PostgreSQL for a
multi-user, online version. On first use, anything already stored in the old per-region files is imported once.
"""
import contextlib
import datetime
import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "regions")          # tests point this at a temporary folder
FILE = "kenyare_flood.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS regions (key TEXT PRIMARY KEY, cfg TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, region TEXT, kind TEXT,
                                   detail TEXT);
CREATE TABLE IF NOT EXISTS candidates (region TEXT NOT NULL, id TEXT NOT NULL, status TEXT, place TEXT, data TEXT,
                                       PRIMARY KEY (region, id));
CREATE TABLE IF NOT EXISTS models (region TEXT NOT NULL, version TEXT NOT NULL, created TEXT, metrics TEXT,
                                   PRIMARY KEY (region, version));
CREATE TABLE IF NOT EXISTS gazetteer (region TEXT NOT NULL, place TEXT NOT NULL, lat REAL, lon REAL, matched TEXT,
                                      PRIMARY KEY (region, place));
CREATE TABLE IF NOT EXISTS decisions (book TEXT NOT NULL, id TEXT NOT NULL, at TEXT, kind TEXT, decision TEXT,
                                      status TEXT, data TEXT, PRIMARY KEY (book, id));
CREATE TABLE IF NOT EXISTS quote_requests (id TEXT PRIMARY KEY, at TEXT, status TEXT, decision_id TEXT, data TEXT);
CREATE INDEX IF NOT EXISTS events_region ON events (region, at);
"""


def path():
    return os.path.join(ROOT, FILE)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _dumps(x):
    return json.dumps(x, default=lambda v: None if v != v else str(v))     # NaN -> null, anything else -> text


@contextlib.contextmanager
def db():
    os.makedirs(ROOT, exist_ok=True)
    c = sqlite3.connect(path(), timeout=15)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        if c.execute("SELECT v FROM meta WHERE k='imported_files'").fetchone() is None:
            _import_files(c)
        yield c
        c.commit()
    finally:
        c.close()


# ------------------------------------------------------------------ regions and the audit trail
def get_region(key):
    with db() as c:
        r = c.execute("SELECT cfg FROM regions WHERE key=?", (key,)).fetchone()
    return json.loads(r["cfg"]) if r else None


def save_region(key, cfg, c=None):
    cfg = {k: v for k, v in cfg.items() if k != "history"}       # history lives in the events table
    if c is not None:
        c.execute("INSERT OR REPLACE INTO regions (key, cfg) VALUES (?, ?)", (key, _dumps(cfg)))
        return
    with db() as c2:
        c2.execute("INSERT OR REPLACE INTO regions (key, cfg) VALUES (?, ?)", (key, _dumps(cfg)))


def region_keys():
    with db() as c:
        return [r["key"] for r in c.execute("SELECT key FROM regions ORDER BY key")]


def delete_region(key):
    with db() as c:
        for t in ("regions", "candidates", "models", "gazetteer"):
            c.execute(f"DELETE FROM {t} WHERE {'key' if t == 'regions' else 'region'}=?", (key,))
        c.execute("INSERT INTO events (at, region, kind, detail) VALUES (?, ?, 'region', 'region deleted')", (now(), key))


def log(region, detail, kind="region", at=None, c=None):
    q = ("INSERT INTO events (at, region, kind, detail) VALUES (?, ?, ?, ?)", (at or now(), region, kind, detail))
    if c is not None:
        c.execute(*q)
        return
    with db() as c2:
        c2.execute(*q)


def events(region=None, limit=500):
    with db() as c:
        if region is None:
            rows = c.execute("SELECT at, region, kind, detail FROM events ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = c.execute("SELECT at, region, kind, detail FROM events WHERE region=? ORDER BY id DESC LIMIT ?",
                             (region, limit))
        return [dict(r) for r in rows]


# ------------------------------------------------------------------ flood-report candidates and approvals
def candidates(region):
    with db() as c:
        return [json.loads(r["data"]) for r in
                c.execute("SELECT data FROM candidates WHERE region=? ORDER BY rowid", (region,))]


def save_candidates(region, cands, c=None):
    def write(cx):
        cx.execute("DELETE FROM candidates WHERE region=?", (region,))
        cx.executemany("INSERT INTO candidates (region, id, status, place, data) VALUES (?, ?, ?, ?, ?)",
                       [(region, x["id"], x.get("status"), x.get("place_name"), _dumps(x)) for x in cands])
    if c is not None:
        write(c)
        return
    with db() as c2:
        write(c2)


# ------------------------------------------------------------------ model versions
def models(region):
    with db() as c:
        rows = c.execute("SELECT version, metrics FROM models WHERE region=?", (region,)).fetchall()
    out = [dict(version=r["version"], **json.loads(r["metrics"])) for r in rows]
    return sorted(out, key=lambda m: int(m["version"][1:]))


def add_model(region, version, metrics, c=None):
    q = ("INSERT OR REPLACE INTO models (region, version, created, metrics) VALUES (?, ?, ?, ?)",
         (region, version, metrics.get("created"), _dumps(metrics)))
    if c is not None:
        c.execute(*q)
        return
    with db() as c2:
        c2.execute(*q)


# ------------------------------------------------------------------ geocoding cache
MISSING = object()


def gaz_get(region, place):
    """(lat, lon, matched), None (looked up before, not found) or MISSING (never looked up)."""
    with db() as c:
        r = c.execute("SELECT lat, lon, matched FROM gazetteer WHERE region=? AND place=?", (region, place)).fetchone()
    if r is None:
        return MISSING
    return None if r["lat"] is None else [r["lat"], r["lon"], r["matched"]]


def gaz_put(region, place, hit, c=None):
    q = ("INSERT OR REPLACE INTO gazetteer (region, place, lat, lon, matched) VALUES (?, ?, ?, ?, ?)",
         (region, place, *(hit[:2] if hit else (None, None)), (hit[2] if hit and len(hit) > 2 else None)))
    if c is not None:
        c.execute(*q)
        return
    with db() as c2:
        c2.execute(*q)


# ------------------------------------------------------------------ decisions and quote requests
def decisions(book):
    with db() as c:
        return [json.loads(r["data"]) for r in
                c.execute("SELECT data FROM decisions WHERE book=? ORDER BY rowid", (book,))]


def put_decisions(book, rows, c=None):
    def write(cx):
        cx.executemany("INSERT OR REPLACE INTO decisions (book, id, at, kind, decision, status, data) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [(book, r["id"], r.get("at"), r.get("kind"), r.get("decision"), r.get("status"), _dumps(r))
                        for r in rows])
    if c is not None:
        write(c)
        return
    with db() as c2:
        write(c2)


def quote_requests():
    with db() as c:
        return [json.loads(r["data"]) for r in c.execute("SELECT data FROM quote_requests ORDER BY rowid")]


def put_quote_request(row, c=None):
    q = ("INSERT OR REPLACE INTO quote_requests (id, at, status, decision_id, data) VALUES (?, ?, ?, ?, ?)",
         (row["id"], row.get("at"), row.get("status"), row.get("decision_id"), _dumps(row)))
    if c is not None:
        c.execute(*q)
        return
    with db() as c2:
        c2.execute(*q)


# ------------------------------------------------------------------ one-time import of the old per-region files
def _import_files(c):
    """Carry over everything stored before the database existed (config.json, candidates.json, models/*/metrics.json,
    gazetteer.json, decisions.csv, quote_requests.csv). The old files are left in place untouched."""
    import pandas as pd
    n = 0
    if os.path.isdir(ROOT):
        for key in sorted(os.listdir(ROOT)):
            d = os.path.join(ROOT, key)
            if not os.path.isdir(d):
                continue
            dp = os.path.join(d, "decisions.csv")              # decisions: every book's folder, region or not
            if os.path.exists(dp) and os.path.getsize(dp):
                put_decisions(key, pd.read_csv(dp).to_dict("records"), c=c)
            cfgp = os.path.join(d, "config.json")
            if not os.path.exists(cfgp):
                continue
            cfg = json.load(open(cfgp))
            for h in cfg.get("history", []):
                log(key, h.get("what", ""), "history (imported)", at=h.get("at"), c=c)
            save_region(key, cfg, c=c)
            cp = os.path.join(d, "candidates.json")
            if os.path.exists(cp):
                save_candidates(key, json.load(open(cp)), c=c)
            md = os.path.join(d, "models")
            if os.path.isdir(md):
                for v in os.listdir(md):
                    mp = os.path.join(md, v, "metrics.json")
                    if os.path.exists(mp):
                        add_model(key, v, json.load(open(mp)), c=c)
            gp = os.path.join(d, "gazetteer.json")
            if os.path.exists(gp):
                for place, hit in json.load(open(gp)).items():
                    gaz_put(key, place, hit, c=c)
            qp = os.path.join(d, "quote_requests.csv")
            if os.path.exists(qp) and os.path.getsize(qp):
                for row in pd.read_csv(qp, dtype=str).to_dict("records"):
                    put_quote_request(row, c=c)
            n += 1
    c.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('imported_files', ?)", (f"{now()}: {n} regions",))
    if n:
        log(None, f"database created; imported {n} region(s) from the old files", "system", c=c)
