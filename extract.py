"""AI step 1: turn unstructured flood reports into structured drainage-failure signals.

Input : sources/*.txt  - one real, citable document per file. First lines are a header:
            title: ...
            url: ...
            date: YYYY-MM-DD
        then a blank line, then the text (news report, county document, drainage study excerpt).
Output: out/signals_raw.json - validated signals with source provenance (no coordinates yet;
        run geocode.py next).

Independence rule: do NOT include the county hotspot-list article itself (or any copy of the
37-hotspot list). The 24 geocoded hotspots are our validation set; if the LLM reads that list,
recall is no longer a test. A guard below warns if a source looks like the list.

LLM provider is pluggable (env LLM_PROVIDER = groq | gemini | anthropic | openai | manual):
  manual  -> writes one prompt per source to out/prompts/; paste each into any chat model and save
             the reply as out/responses/<source_id>.json, then re-run. Works with no API key.

usage: python extract.py [sources_dir]
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sources")
OUT = os.path.join(HERE, "out")

MECHANISMS = ["drainage_blockage", "inadequate_drainage_capacity", "encroachment_on_drainage",
              "impervious_runoff", "river_overflow", "unknown"]
PLACE_TYPES = ["estate", "road", "junction", "neighbourhood", "landmark", "river_section", "other"]

SCHEMA = {
    "type": "object",
    "required": ["signals"],
    "properties": {"signals": {"type": "array", "items": {
        "type": "object",
        "required": ["place_name", "place_type", "mechanism", "severity", "evidence_quote", "confidence"],
        "properties": {
            "place_name": {"type": "string", "description": "Most specific place named in the text"},
            "place_type": {"enum": PLACE_TYPES},
            "mechanism": {"enum": MECHANISMS},
            "severity": {"enum": [1, 2, 3]},
            "event_date": {"type": ["string", "null"], "description": "YYYY-MM if stated"},
            "evidence_quote": {"type": "string", "description": "Exact sentence copied from the text"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }}}},
}

PROMPT = """You are extracting flood evidence for a catastrophe model of Nairobi, Kenya.

From the document below, list every specific place in Nairobi that the text says FLOODED or is
FLOOD-PRONE. For each place return:
- place_name: the most specific place named (estate, road, junction, neighbourhood). Copy the name as written.
- place_type: one of {place_types}
- mechanism: why it floods, according to the text. One of {mechanisms}.
  Use river_overflow only if the text attributes it to a river bursting its banks; use unknown if the text does not say.
- severity: 1 = roads or open ground waterlogged; 2 = homes or businesses flooded or damaged;
  3 = deaths, displacement, or prolonged/repeated flooding.
- event_date: YYYY-MM if the text gives one, else null.
- evidence_quote: ONE sentence copied EXACTLY, word for word, from the text that supports this entry.
- confidence: 0-1, how clearly the text ties flooding to THIS specific place.

Rules: only use what the text says; do not add places from general knowledge; skip places outside
Nairobi. If nothing qualifies, return an empty list.

Respond with JSON only, in this shape: {{"signals": [ ... ]}}

DOCUMENT (source_id={source_id}):
<<<
{text}
>>>"""

HOTSPOT_NAMES = ["Kiambiu", "Dandora", "Kariobangi", "Kayole", "Komarock", "Njiru", "Ruai", "Mwiki", "Donholm",
                 "Tassia", "Fedha", "Madaraka", "Nairobi West", "Lang'ata", "Kawangware", "Kangemi", "Lavington",
                 "Westlands", "Parklands", "Kitisuru", "Kileleshwa", "Chiromo", "Mathare", "Kibera"]


# ------------------------------------------------------------------ sources
def read_source(path):
    raw = open(path, encoding="utf-8").read()
    head, _, body = raw.partition("\n\n")
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", head, flags=re.M))
    sid = os.path.splitext(os.path.basename(path))[0]
    return dict(source_id=sid, title=meta.get("title", ""), url=meta.get("url", ""),
                date=meta.get("date", ""), text=body.strip())


def looks_like_hotspot_list(text, threshold=10):
    hits = [n for n in HOTSPOT_NAMES if n.lower() in text.lower()]
    return len(hits) >= threshold, hits


# ------------------------------------------------------------------ LLM providers (see llm.py)
from llm import ManualMode, parse_json


def call_llm(prompt):
    import llm
    return llm.complete(prompt, json_mode=True)


# ------------------------------------------------------------------ validation
def _norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def validate(signals, source):
    """Keep only well-formed signals whose evidence_quote appears verbatim in the source.
    Returns (kept, rejected-with-reason). This is the main guard against the LLM inventing evidence."""
    kept, rejected = [], []
    body = _norm(source["text"])
    for s in signals:
        reason = None
        missing = [k for k in SCHEMA["properties"]["signals"]["items"]["required"] if k not in s]
        if missing:
            reason = f"missing {missing}"
        elif s["mechanism"] not in MECHANISMS:
            reason = f"bad mechanism {s['mechanism']}"
        elif s["severity"] not in (1, 2, 3):
            reason = f"bad severity {s['severity']}"
        elif not (0 <= float(s["confidence"]) <= 1):
            reason = "confidence out of range"
        elif _norm(s["evidence_quote"].strip(' ."')) not in body:
            reason = "evidence_quote not found verbatim in source"
        elif _norm(s["place_name"]) not in body:
            reason = "place_name not in source"
        if reason:
            rejected.append({**s, "reject_reason": reason})
        else:
            kept.append({**s, "source_id": source["source_id"], "source_title": source["title"],
                         "source_url": source["url"], "source_date": source["date"]})
    return kept, rejected


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SOURCES, "*.txt")))
    if not files:
        sys.exit(f"No sources in {SOURCES}. Add one .txt per document (see header format in this file).")
    all_kept, all_rej, pending = [], [], []
    for f in files:
        src = read_source(f)
        is_list, hits = looks_like_hotspot_list(src["text"])
        if is_list:
            print(f"SKIP {src['source_id']}: names {len(hits)} of the 24 validation hotspots - looks like the "
                  f"county list itself. Excluded to keep validation independent.")
            continue
        prompt = PROMPT.format(place_types=PLACE_TYPES, mechanisms=MECHANISMS,
                               source_id=src["source_id"], text=src["text"])
        resp_path = os.path.join(OUT, "responses", src["source_id"] + ".json")
        try:
            if os.path.exists(resp_path):
                reply = open(resp_path, encoding="utf-8").read()
            else:
                reply = call_llm(prompt)
                os.makedirs(os.path.dirname(resp_path), exist_ok=True)
                open(resp_path, "w", encoding="utf-8").write(reply)   # cache: re-runs are free and reproducible
        except ManualMode:
            os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
            open(os.path.join(OUT, "prompts", src["source_id"] + ".txt"), "w", encoding="utf-8").write(prompt)
            pending.append(src["source_id"])
            continue
        kept, rej = validate(parse_json(reply).get("signals", []), src)
        all_kept += kept
        all_rej += rej
        print(f"{src['source_id']}: {len(kept)} signals kept, {len(rej)} rejected")
    json.dump(dict(signals=all_kept, rejected=all_rej), open(os.path.join(OUT, "signals_raw.json"), "w"), indent=2)
    if pending:
        print(f"\nManual mode: {len(pending)} prompts written to out/prompts/. Paste each into a chat model, save the "
              f"JSON reply as out/responses/<source_id>.json, then re-run.")
    print(f"\nTotal: {len(all_kept)} kept, {len(all_rej)} rejected -> out/signals_raw.json")


if __name__ == "__main__":
    main()
