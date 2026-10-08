"""Visual identity: Kenya Re's public website palette and fonts (kenyare.co.ke theme CSS: --knre-global-color #C30D35,
--knre-secondary-color #041D3B, --knre-light-color #ECF0F4; headings Archivo, body Roboto). The Kenya Re mark (assets/,
cut from the website logo and split into its three colour pieces for the loading animation) is used for this Kenya Re
hackathon; the sidebar still says this is a prototype, not a Kenya Re product.

Chart colours were checked with the dataviz palette validator (lightness band, chroma, colour-blind separation):
brand navy #041D3B is too dark/grey to be a data series, so it is used for text, headings and the one emphasised
line; series use SERIES in fixed order. Status colours (risk bands, claim verdicts, hotspot hit/miss) are kept
separate on purpose: green / amber / red must keep meaning good / warning / bad.
"""
import base64, os
import plotly.graph_objects as go
import plotly.io as pio

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MARK_PATH = os.path.join(ASSETS, "mark.png")


def _uri(name):
    """A logo piece as a data URI (embedded in the page, so nothing extra has to be served)."""
    with open(os.path.join(ASSETS, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

CRIMSON, NAVY, LIGHT, BODY_GREY = "#C30D35", "#041D3B", "#ECF0F4", "#696F6F"
BLUE, LIGHT_BLUE, GOLD = "#1F5AA6", "#5FA8E0", "#B97E00"
SERIES = [BLUE, CRIMSON, LIGHT_BLUE, GOLD]          # validated categorical order
INK, MUTED, GRID = NAVY, "#5B6470", "#E3E8EE"
NEUTRAL_MARK = "#B9C0C8"                            # 'other assets' dots on maps
SEQ_CRIMSON = [[0, "#F7E1E6"], [0.5, "#E0627F"], [1, "#8E0A27"]]   # one hue, light -> dark
SEQ_BLUE = [[0, "#EEF3F9"], [0.5, "#6F9BD1"], [1, "#0F3B73"]]

FONT_BODY, FONT_HEAD = "Roboto, Arial, sans-serif", "Archivo, Roboto, Arial, sans-serif"

_t = go.layout.Template(pio.templates["simple_white"])
_t.layout.update(font=dict(family=FONT_BODY, color=INK, size=13), colorway=SERIES,
                 title=dict(font=dict(family=FONT_HEAD, color=NAVY, size=16)),
                 xaxis=dict(linecolor=GRID, tickcolor=GRID, gridcolor=GRID),
                 yaxis=dict(linecolor=GRID, tickcolor=GRID, gridcolor=GRID),
                 hoverlabel=dict(font=dict(family=FONT_BODY)))
pio.templates["kenyare"] = _t
TEMPLATE = "kenyare"

# small CSS touches the Streamlit theme (.streamlit/config.toml) can't express
CSS = f"""<style>
:root {{ --kre-ink: {NAVY}; --kre-accent: {CRIMSON}; --kre-canvas: #F7F8FA; --kre-line: {GRID}; }}
[data-testid="stMain"] {{ background: var(--kre-canvas); }}
[data-testid="stMainBlockContainer"] {{ max-width: 1500px; padding: 2rem 2.5rem 4rem; }}
h1, h2, h3, h4 {{ letter-spacing: 0; color: var(--kre-ink); }}
h1 {{ border: 0; padding: 0; display: block; }}
h2 {{ font-size: 1.35rem; }}
h3 {{ font-size: 1.1rem; }}
[data-testid="stMarkdownContainer"] p {{ line-height: 1.55; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-family: Archivo, Roboto, sans-serif; font-size: 1.85rem; }}
[data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; }}
/* sidebar navigation: grouped page links (st.page_link) in keyed containers - .st-key-* classes are Streamlit's
   supported styling hook; the current page sits in .st-key-kre_nav_on */
[data-testid="stSidebar"] > div:first-child {{ background: #F2F4F7; }}
[data-testid="stSidebarUserContent"] {{ background: linear-gradient(180deg, {NAVY} 0%, #0A2A4F 100%); border-radius: 16px;
  margin: .75rem; padding: 1.25rem 1rem; box-shadow: 0 4px 18px rgba(4, 29, 59, .25); }}
.kre-nav-group {{ font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: #7F93AD; font-weight: 700;
  margin: .6rem 0 .1rem .3rem; }}
.st-key-kre_nav, .st-key-kre_nav [data-testid="stVerticalBlock"] {{ gap: .3rem; }}
.st-key-kre_nav a[data-testid="stPageLink-NavLink"] {{ padding: .55rem .8rem; border-radius: 6px; background: transparent; }}
.st-key-kre_nav a[data-testid="stPageLink-NavLink"]:hover {{ background: rgba(255, 255, 255, .08); }}
.st-key-kre_nav a[data-testid="stPageLink-NavLink"] span,
.st-key-kre_nav a[data-testid="stPageLink-NavLink"] p {{ color: #E6ECF3 !important; font-size: .98rem; }}
.st-key-kre_nav a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {{ color: #9FB3CC !important; font-size: 1.25rem; }}
.st-key-kre_nav_on a[data-testid="stPageLink-NavLink"] {{ background: {CRIMSON} !important;
  box-shadow: none; }}
.st-key-kre_nav_on a[data-testid="stPageLink-NavLink"] span, .st-key-kre_nav_on a[data-testid="stPageLink-NavLink"] p,
.st-key-kre_nav_on a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {{ color: #FFFFFF !important; font-weight: 700; }}
/* sidebar expanders (analyst controls): light heading on the navy panel, white body with navy text */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{ color: #E6ECF3 !important; fill: #E6ECF3; }}
[data-testid="stSidebar"] [data-testid="stExpander"] details {{ border-color: rgba(255, 255, 255, .18); }}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {{ background: #FFFFFF; border-radius: 0 0 8px 8px; }}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stMarkdown"],
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] label,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] label p,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] label span {{ color: {NAVY} !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #9FB3CC; }}
/* navy carries the layout, crimson is the accent */
[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {GRID}; border-top: 2px solid {NAVY};
  border-radius: 6px; padding: .8rem 1rem; }}
[data-testid="stButton"] button {{ border-radius: 6px; font-weight: 600; min-height: 2.6rem; }}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {{ border-radius: 6px; }}
[data-testid="stTab"][aria-selected="true"] {{ border-bottom: 3px solid {NAVY} !important; }}
[data-testid="stTab"][aria-selected="true"]::after, [data-testid="stTab"][aria-selected="true"]::before {{ background: {NAVY} !important; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {NAVY} !important; font-weight: 700; }}
[data-baseweb="tab-highlight"] {{ background-color: {NAVY} !important; }}
[data-testid="stMain"] [data-testid="stExpander"] summary {{ background: #F4F7FB; }}
[data-testid="stMain"] [data-testid="stExpander"] summary:hover p {{ color: {NAVY}; }}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{ background: rgba(255, 255, 255, .06); }}
/* chat: user turns as right-aligned bubbles, assistant turns plain (ChatGPT / Claude style) */
[data-testid="stChatMessage"] {{ background: transparent; max-width: 820px; margin: 0 auto; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{ justify-content: flex-end !important; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] {{ display: none; }}
[aria-label="Chat message from user"] {{ background: {LIGHT}; border-radius: 18px; padding: .55rem 1rem;
  flex: 0 1 auto !important; width: fit-content; max-width: 75%; margin-left: auto !important; margin-right: 0 !important; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid^="stChatMessageAvatar"] {{
  background: {CRIMSON}; color: #fff; border: none; }}
[data-testid="stChatInput"] {{ max-width: 820px; margin: 0 auto; }}
/* citations in assistant answers (NotebookLM-style): a numbered chip; hover or click shows where it came from */
.kre-cite {{ position: relative; display: inline-flex; align-items: center; justify-content: center; min-width: 17px;
  height: 17px; padding: 0 5px; margin-left: 3px; border-radius: 9px; background: #DDE5F0; color: {NAVY};
  font-size: 10.5px; font-weight: 700; line-height: 1; vertical-align: 0.35em; cursor: pointer; }}
.kre-cite:hover, .kre-cite:focus {{ background: {NAVY}; color: #fff; outline: none; }}
.kre-pop {{ display: none; position: absolute; left: -8px; top: calc(100% + 7px); z-index: 1000; width: min(380px, 70vw);
  background: #fff; color: #1D2430; border: 1px solid {GRID}; border-left: 3px solid {CRIMSON}; border-radius: 10px;
  box-shadow: 0 10px 28px rgba(4, 29, 59, .18); padding: 10px 12px; font-size: 12.5px; font-weight: 400;
  line-height: 1.5; white-space: normal; text-align: left; }}
.kre-pop b {{ color: {NAVY}; display: block; margin-bottom: 2px; }}
.kre-cite:hover .kre-pop, .kre-cite:focus .kre-pop {{ display: block; }}
[data-testid="stChatMessage"], [data-testid="stChatMessageContent"] {{ overflow: visible !important; }}
</style>"""

SIDEBAR_BRAND = (f"<div style='display:flex;align-items:center;gap:.8rem'>"
                 f"<div style='background:#fff;border-radius:12px;width:48px;height:48px;display:flex;"
                 f"align-items:center;justify-content:center;flex:none;box-shadow:0 2px 8px rgba(0,0,0,.25)'>"
                 f"<img src='{_uri('mark.png')}' alt='Kenya Re' style='width:40px;height:40px'></div>"
                 f"<div style='font-family:Archivo,sans-serif;font-weight:700;font-size:1.15rem;color:#fff;line-height:1.15'>"
                 f"Nairobi Flood<br><span style='color:#FF6B86'>Risk Workbench</span></div></div>"
                 f"<div style='font-size:.72rem;color:#9FB3CC;margin-top:.6rem'>Kenya Re AI4I Hackathon 2026 · Team A · "
                 "prototype, not a Kenya Re product</div>"
                 f"<hr style='border:none;border-top:1px solid rgba(255,255,255,.15);margin:1rem 0 .5rem'>")


def page_header(title, subtitle_html, intro=None):
    """Compact page title: the name, an optional one-line purpose, and the data caveats folded into one line that
    opens on click (kept visible on every page, but small)."""
    short = "Prototype · synthetic portfolio · estimated flood map · assumed terms"
    return (f"<div style='margin:0 0 .9rem'>"
            f"<div style='display:flex;align-items:center;gap:.6rem'><span style='width:5px;height:1.7rem;"
            f"background:{CRIMSON};border-radius:3px'></span><span style='font-family:Archivo,sans-serif;font-size:1.75rem;"
            f"font-weight:700;color:{NAVY};line-height:1.1'>{title}</span></div>"
            + (f"<div style='color:#3A4554;font-size:.92rem;margin:.35rem 0 0 .9rem'>{intro}</div>" if intro else "")
            + f"<details style='margin:.35rem 0 0 .9rem;font-size:.76rem;color:{MUTED}'><summary style='cursor:pointer;"
            f"list-style:none'>ⓘ {short} <span style='text-decoration:underline'>what this means</span></summary>"
            f"<div style='margin-top:.3rem;max-width:760px;line-height:1.5'>{subtitle_html}</div></details></div>")


# Non-blocking progress toast: visible during long model runs without obscuring the controls or results.
_LOADER_CSS = """<style>
.kre-loader { position: fixed; inset: auto 1rem 1rem auto; z-index: 999990; display: none; flex-direction: row;
  align-items: center; gap: 10px; max-width: calc(100vw - 2rem); padding: 10px 14px; background: #fff;
  border: 1px solid __GRID__; border-left: 4px solid __CRIMSON__; border-radius: 6px;
  box-shadow: 0 4px 18px rgba(4, 29, 59, .12); opacity: 0; pointer-events: none; }
body:has([data-testid="stStatusWidget"]) .kre-loader { display: flex; animation: kre-in .3s ease .6s forwards; }
@keyframes kre-in { to { opacity: 1; } }
body:has([data-testid="stChatInput"]) .kre-loader { display: none !important; }   /* assistant: has its own "Thinking..." */
body:has(.kre-agent-live) .kre-loader { display: none !important; }   /* review agent: shows its own live steps */
.kre-mark { position: relative; flex: none; width: 26px; height: 32px; animation: none; }
.kre-mark img { position: absolute; inset: 0; width: 100%; height: 100%; }
.kre-mark img { animation: none !important; }
@keyframes kre-grey  { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(-20px, -18px) rotate(-8deg); } }
@keyframes kre-red   { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(20px, -10px) rotate(7deg); } }
@keyframes kre-navy  { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(-8px, 22px) rotate(-4deg); } }
@keyframes kre-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(.92); } }
.kre-loader-text { font-family: Archivo, Roboto, sans-serif; font-weight: 600; font-size: .82rem; color: __NAVY__; }
.kre-loader-sub { display: none; }
@media (prefers-reduced-motion: reduce) {
  .kre-mark, .kre-mark img { animation: none !important; }
}
@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3rem; }
  .kre-loader { inset: auto .75rem .75rem auto; }
}
</style>""".replace("__NAVY__", NAVY).replace("__GRID__", GRID).replace("__CRIMSON__", CRIMSON)


def loader(text="Running the flood model", sub="hazard, damage and loss for every building"):
    """The overlay markup (CSS + the three stacked pieces). Put it on the page once per run."""
    return (_LOADER_CSS + "<div class='kre-loader' role='status' aria-live='polite'><div class='kre-mark'>"
            f"<img class='kre-grey' src='{_uri('mark_grey.png')}' alt=''>"
            f"<img class='kre-navy' src='{_uri('mark_navy.png')}' alt=''>"
            f"<img class='kre-red' src='{_uri('mark_red.png')}' alt=''></div>"
            f"<div class='kre-loader-text'>{text}…</div><div class='kre-loader-sub'>{sub}</div></div>")



# headline number cards: a responsive grid (cards wrap to a new row instead of squeezing), label + value + sub-line all
# inside the card, value sized to fit. One card may be marked key (crimson edge) - the figure the page is about.
_KPI_CSS = """<style>
.kre-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 14px; margin: .3rem 0 1.35rem; }
.kre-kpi { background: #FFFFFF; border: 1px solid __GRID__; border-top: 2px solid __NAVY__; border-radius: 6px;
  padding: 18px 18px 16px; min-width: 0; }
.kre-kpi.key { border-top-color: __CRIMSON__; background: #FFF9FA; }
.kre-kpi .l { font-size: .8rem; color: #4A5565; line-height: 1.3; }
.kre-kpi .v { font-family: Archivo, Roboto, sans-serif; font-weight: 600; color: __NAVY__; line-height: 1.1;
  font-size: clamp(1.3rem, 1.05rem + 0.9vw, 1.85rem); margin: .35rem 0 .3rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; }
.kre-kpi .s { font-size: .76rem; color: #5B6470; line-height: 1.45; }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", NAVY).replace("__CRIMSON__", CRIMSON)


def kpis(items, min_px=215):
    """items: list of dicts {label, value, sub?, key?}; sub = a string or a list of lines. Returns the cards' HTML."""
    import html as _h
    cards = "".join(f"<div class='kre-kpi{' key' if it.get('key') else ''}' title='{_h.escape(it['label'])}'>"
                    f"<div class='l'>{_h.escape(it['label'])}</div><div class='v'>{_h.escape(str(it['value']))}</div>"
                    + (f"<div class='s'>{'<br>'.join(_h.escape(x) for x in (it['sub'] if isinstance(it['sub'], (list, tuple)) else [it['sub']]))}</div>"
                       if it.get("sub") else "") + "</div>"
                    for it in items)
    return _KPI_CSS + (f"<div class='kre-kpis' style='grid-template-columns:repeat(auto-fit,minmax({min_px}px,1fr))'>"
                       f"{cards}</div>")


# structured briefing card (briefing.py): stance + headline, drivers up / down, next steps, questions, caveat
_BRIEF_CSS = """<style>
.kre-brief { border: 1px solid __GRID__; border-radius: 14px; padding: 16px 18px 14px; background: #fff;
  box-shadow: 0 2px 10px rgba(4, 29, 59, .06); margin: .2rem 0 1rem; }
.kre-brief .top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.kre-brief .pill { padding: 4px 11px; border-radius: 999px; font-weight: 700; font-size: .8rem; }
.kre-brief .pill.good { background: #E3F1E3; color: #1D6B1D; }
.kre-brief .pill.warn { background: #FFF1D6; color: #8A5A00; }
.kre-brief .pill.bad { background: #F9D6D6; color: #A11D1D; }
.kre-brief .src { font-size: .72rem; color: #5B6470; }
.kre-brief .hl { font-family: Archivo, Roboto, sans-serif; font-weight: 600; font-size: 1.12rem; color: __NAVY__;
  line-height: 1.35; margin: .65rem 0 .8rem; }
.kre-brief .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; align-items: start; }
.kre-brief .drvs { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; margin-bottom: 10px; align-items: start; }
.kre-brief details { border-radius: 10px; background: #F4F7FB; }
.kre-brief summary { list-style: none; cursor: pointer; padding: 8px 10px; display: flex; align-items: center; gap: 8px;
  font-size: .86rem; font-weight: 600; color: __NAVY__; }
.kre-brief summary::-webkit-details-marker { display: none; }
.kre-brief summary::after { content: '▾'; margin-left: auto; color: #8A94A3; font-size: .75rem; }
.kre-brief details[open] summary::after { content: '▴'; }
.kre-brief .ic { flex: none; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: .7rem; color: #fff; }
.kre-brief .up .ic { background: __CRIMSON__; }
.kre-brief .down .ic { background: __BLUE__; }
.kre-brief .drv.up { box-shadow: inset 3px 0 0 __CRIMSON__; }
.kre-brief .drv.down { box-shadow: inset 3px 0 0 __BLUE__; }
.kre-brief details > span, .kre-brief details > ul { display: block; font-size: .8rem; color: #3A4554; line-height: 1.4;
  padding: 0 10px 9px 40px; margin: 0; }
.kre-brief details > ul { padding-left: 1.6rem; }
.kre-brief .n { background: __NAVY__; color: #fff; border-radius: 999px; font-size: .7rem; padding: 1px 7px; }
.kre-brief li { font-size: .82rem; color: #1D2430; line-height: 1.45; margin-bottom: 3px; }
.kre-brief .cav { font-size: .74rem; color: #5B6470; margin-top: .6rem; }
</style>""".replace("__GRID__", GRID).replace("__NAVY__", NAVY).replace("__CRIMSON__", CRIMSON).replace("__BLUE__", BLUE)


def briefing_html(b, audience="underwriter", show_top=True):
    """Compact card: stance + headline, then the drivers as small cards (click one for the detail) and the next steps /
    questions folded away with a count, so the underwriter takes it in at a glance."""
    import html as _h
    e = _h.escape
    order = [x for x in b["drivers"] if x["effect"] == "raises"] + [x for x in b["drivers"] if x["effect"] != "raises"]
    drv = "".join(f"<details class='drv {'up' if x['effect'] == 'raises' else 'down'}'><summary>"
                  f"<span class='ic'>{'▲' if x['effect'] == 'raises' else '▼'}</span>{e(x['factor'])}</summary>"
                  f"<span>{e(x['detail'])}</span></details>" for x in order)
    fold = lambda title, xs: (f"<details><summary>{title} <span class='n'>{len(xs)}</span></summary><ul>"
                              + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul></details>") if xs else ""
    src = ("AI-written · numbers checked ✓" + (f" · {b['dropped']} unsupported item(s) removed" if b.get("dropped") else "")
           if b["source"] == "ai" else "Rule-based summary")
    ask = "Ask the broker" if audience == "underwriter" else "Worth checking"
    return _BRIEF_CSS + (
        f"<div class='kre-brief'>"
        + (f"<div class='top'><span class='pill {b['tone']}'>{e(b['stance'])}</span>"
           f"<span class='src'>{e(src)}</span></div><div class='hl'>{e(b['headline'])}</div>" if show_top else
           f"<div class='top'><b style='color:{NAVY};font-size:.9rem'>Why</b><span class='src'>{e(src)}</span></div>")
        + f"<div class='drvs'>{drv}</div>"
        f"<div class='grid'>{fold('Next steps', b['actions'])}{fold(ask, b['questions'])}</div>"
        f"<div class='cav'>ⓘ {e(b['caveat'])}</div></div>")
