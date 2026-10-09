"""Flood-report scraper for the model workspace: finds recent news about flooding in a city, reads each article, and
returns CANDIDATE flood places for an underwriter to approve. Nothing reaches the model until it is approved.

  discover(city)                 news search -> article links (Bing News RSS; GDELT as a second source when it answers)
  harvest(city, ..., call)       fetch each new article -> LLM extraction with the same checks as extract.py (the
                                 quote must appear word for word; forecasts and warnings are rejected) -> repeated
                                 news merged per place -> coordinates (OpenStreetMap) -> candidates

Repeated news: the same story syndicated by several outlets is dropped by near-identical title; several articles about
the same place are merged into one candidate that lists every source (so one storm doesn't count ten times).
Article text is saved in the region's sources/ folder only (kept out of git, like sources/*.txt).
"""
import difflib
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

UA = "Mozilla/5.0 (Kenya Re AI4I hackathon flood research; contact via the team)"
FLOOD_WORDS = re.compile(r"\bflood(s|ed|ing|waters?)?\b|\bsubmerged\b|\binundat", re.I)
MIN_CHARS = 600


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()


def _canonical(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc.lower().removeprefix("www."), p.path.rstrip("/"), "", ""))


def _bing(query):
    x = _get("https://www.bing.com/news/search?" + urllib.parse.urlencode({"q": query, "format": "rss"}))
    out = []
    for it in ET.fromstring(x).findall(".//item"):
        link = it.findtext("link") or ""
        m = re.search(r"[?&]url=([^&]+)", link)
        url = urllib.parse.unquote(m.group(1)) if m else link
        try:
            date = parsedate_to_datetime(it.findtext("pubDate")).date().isoformat()
        except Exception:
            date = ""
        out.append(dict(url=url, title=(it.findtext("title") or "").strip(), date=date, found_by="Bing News"))
    return out


def _gdelt(query):
    q = urllib.parse.urlencode({"query": query + " sourcelang:english", "mode": "artlist", "format": "json",
                                "maxrecords": 50, "timespan": "3months", "sort": "datedesc"})
    r = json.loads(_get("https://api.gdeltproject.org/api/v2/doc/doc?" + q))
    return [dict(url=a["url"], title=a.get("title", ""), date=(a.get("seendate") or "")[:8], found_by="GDELT")
            for a in r.get("articles", [])]


def dedupe(items, known=()):
    """Drop links already seen (by canonical URL) and syndicated copies (titles >= 85% alike). Returns (kept, n_dropped)."""
    seen = {_canonical(u) for u in known}
    kept, titles, dropped = [], [], 0
    for it in items:
        c, t = _canonical(it["url"]), re.sub(r"\W+", " ", it["title"].lower()).strip()
        if c in seen or any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.85 for x in titles if t):
            dropped += 1
            continue
        seen.add(c)
        titles.append(t)
        kept.append(it)
    return kept, dropped


def discover(city, known=(), log=print, use_gdelt=True):
    """Article links about flooding in `city`, newest first, with repeats removed."""
    items = []
    for q in (f"{city} floods", f"{city} flooding", f"{city} flooded homes"):
        try:
            got = _bing(q)
            items += got
            log(f"Bing News '{q}': {len(got)} articles")
        except Exception as e:
            log(f"Bing News '{q}' failed: {type(e).__name__}")
    if use_gdelt:
        try:
            got = _gdelt(f'"{city}" (flood OR flooding OR floods)')
            items += got
            log(f"GDELT: {len(got)} articles")
        except Exception as e:                        # GDELT often rate-limits (HTTP 429): the scrape goes on without it
            log(f"GDELT unavailable ({str(e)[:40]}) - continuing with Bing News")
    items.sort(key=lambda x: x["date"], reverse=True)
    kept, dropped = dedupe(items, known)
    log(f"{len(kept)} new articles after removing {dropped} repeats and ones already read")
    return kept


def fetch_text(url):
    from fetch_sources import text_of
    html = _get(url, timeout=30).decode("utf-8", "replace")
    return text_of(html)


def _prompt(city, source_id, text):
    import extract as ex
    p = ex.PROMPT.replace("Nairobi, Kenya", f"{city}, Kenya").replace("Nairobi", city)
    return p.format(place_types=ex.PLACE_TYPES, mechanisms=ex.MECHANISMS, source_id=source_id, text=text[:24000])


def harvest(city, call, sources_dir, known_urls=(), max_articles=12, geocoder=None, log=print, articles=None,
            is_validation_list=None):
    """Read new articles and return (candidates, stats). call(prompt) -> str is the LLM. geocoder(place) -> (lat, lon)
    or None. articles: skip discovery and use these (for tests). is_validation_list(text) -> True skips a source that
    looks like the held-out validation list itself."""
    import extract as ex
    from llm import parse_json
    os.makedirs(sources_dir, exist_ok=True)
    arts = articles if articles is not None else discover(city, known_urls, log)
    stats = dict(found=len(arts), read=0, too_short=0, off_topic=0, kept=0, rejected=0, reasons={})
    signals = []
    for a in arts[:max_articles]:
        sid = hashlib.sha1(_canonical(a["url"]).encode()).hexdigest()[:12]
        try:
            text = a.get("text") or fetch_text(a["url"])
        except Exception as e:
            log(f"could not read {a['url'][:70]} ({type(e).__name__})")
            continue
        if len(text) < MIN_CHARS:
            stats["too_short"] += 1
            continue
        if city.lower() not in text.lower() or not FLOOD_WORDS.search(text):
            stats["off_topic"] += 1
            continue
        if is_validation_list and is_validation_list(text):
            log(f"skipped '{a['title'][:60]}': it looks like the validation list itself")
            continue
        open(os.path.join(sources_dir, sid + ".txt"), "w", encoding="utf-8").write(text)
        src = dict(source_id=sid, title=a["title"], url=a["url"], date=a.get("date", ""), text=text)
        stats["read"] += 1
        try:
            reply = call(_prompt(city, sid, text))
            got = parse_json(reply).get("signals", [])
        except Exception as e:
            log(f"extraction failed for '{a['title'][:50]}' ({type(e).__name__})")
            continue
        kept, rej = ex.validate(got, src)
        stats["kept"] += len(kept)
        stats["rejected"] += len(rej)
        for r in rej:
            k = r["reject_reason"].split(" (")[0]
            stats["reasons"][k] = stats["reasons"].get(k, 0) + 1
        signals += kept
        log(f"'{a['title'][:60]}': {len(kept)} flood places, {len(rej)} rejected")
    return merge(signals, geocoder, log), stats


def place_key(name):
    """'Port Reitz creeks' and 'Port Reitz Creek' are the same place: lower case, no punctuation, no plural s."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w
                    for w in re.sub(r"\W+", " ", str(name).lower()).split())


def merge(signals, geocoder=None, log=print):
    """One candidate per place: every source and quote listed, the highest severity, the mean confidence."""
    by = {}
    for s in signals:
        k = place_key(s["place_name"])
        by.setdefault(k, []).append(s)
    out = []
    for k, rows in by.items():
        lat = lon = None
        matched = ""
        if geocoder:
            try:
                g = geocoder(rows[0]["place_name"])
                if g:
                    lat, lon, matched = g[0], g[1], (g[2] if len(g) > 2 else "")
            except Exception:
                pass
        out.append(dict(id=hashlib.sha1(k.encode()).hexdigest()[:10], place_name=rows[0]["place_name"],
                        place_type=rows[0]["place_type"], mechanism=rows[0]["mechanism"],
                        severity=max(int(r["severity"]) for r in rows),
                        confidence=round(sum(float(r["confidence"]) for r in rows) / len(rows), 2),
                        n_sources=len({r["source_url"] for r in rows}), lat=lat, lon=lon, found_as=matched,
                        status="pending",
                        signals=rows))
    if geocoder:
        log(f"{sum(c['lat'] is not None for c in out)} of {len(out)} places found on the map")
    return sorted(out, key=lambda c: (-c["n_sources"], -c["severity"]))
