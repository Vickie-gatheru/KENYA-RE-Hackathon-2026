"""Download the reports listed in data/sources.csv into sources/<source_id>.txt.  RUN ON YOUR LAPTOP.

Uses trafilatura for clean article text if installed (pip install trafilatura), otherwise a simple
<p>-tag extractor. Paywalled / JavaScript-only pages may come back short; those are skipped and listed
so you can paste the text in by hand (keep the same header format).

Deliberately EXCLUDED: the articles reproducing the government's 37-hotspot list - that list is our
validation set. extract.py also auto-skips any text naming >= 10 of the 24 hotspots.

usage: python fetch_sources.py
"""
import csv, os, re, urllib.request
from html.parser import HTMLParser

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sources")
UA = "Mozilla/5.0 (hackathon research; student project)"
MIN_CHARS = 600


class P(HTMLParser):
    def __init__(self):
        super().__init__(); self.buf, self.inp, self.skip = [], False, 0
    def handle_starttag(self, t, a):
        if t in ("script", "style", "nav", "footer"): self.skip += 1
        if t == "p": self.inp = True; self.buf.append("\n")
    def handle_endtag(self, t):
        if t in ("script", "style", "nav", "footer"): self.skip = max(0, self.skip - 1)
        if t == "p": self.inp = False
    def handle_data(self, x):
        if self.inp and not self.skip: self.buf.append(x)


def text_of(html):
    try:
        import trafilatura
        t = trafilatura.extract(html)
        if t: return t
    except ImportError:
        pass
    p = P(); p.feed(html)
    return re.sub(r"\n\s*\n+", "\n\n", "".join(p.buf)).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "sources.csv"), encoding="utf-8")))
    short = []
    for r in rows:
        path = os.path.join(OUT, r["source_id"] + ".txt")
        if os.path.exists(path):
            print(f"have   {r['source_id']}"); continue
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
            body = text_of(html)
        except Exception as e:
            body = ""; print(f"ERROR  {r['source_id']}: {e}")
        if len(body) < MIN_CHARS:
            short.append(r); print(f"SHORT  {r['source_id']} ({len(body)} chars) - paste text manually"); continue
        date = re.search(r"(20\d\d)[-/](\d\d)[-/](\d\d)", r["url"])
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"title: {r['title']}\nurl: {r['url']}\ndate: {'-'.join(date.groups()) if date else ''}\n\n{body}\n")
        print(f"saved  {r['source_id']} ({len(body)} chars)")
    print(f"\n{len(rows) - len(short)} saved, {len(short)} need manual text: " + ", ".join(r["source_id"] for r in short))


if __name__ == "__main__":
    main()
