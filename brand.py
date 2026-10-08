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
h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
h1 {{ border-bottom: 3px solid {CRIMSON}; padding-bottom: .25rem; display: inline-block; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-family: Archivo, Roboto, sans-serif; font-size: 1.85rem; }}
[data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; }}
/* sidebar navigation: icon menu with a filled pill for the current page (no radio circles) */
[data-testid="stSidebar"] > div:first-child {{ background: #F2F4F7; }}
[data-testid="stSidebarUserContent"] {{ background: linear-gradient(180deg, {NAVY} 0%, #0A2A4F 100%); border-radius: 16px;
  margin: .75rem; padding: 1.25rem 1rem; box-shadow: 0 4px 18px rgba(4, 29, 59, .25); }}
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
[data-testid="stSidebar"] [data-testid="stRadio"] {{ width: 100% !important; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: .25rem; width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] > div, [data-testid="stSidebar"] [role="radiogroup"] > label {{ width: 100%; }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{ width: 100%; padding: .7rem .9rem; border-radius: 10px; margin: 0;
  transition: background .15s ease; cursor: pointer; }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{ font-size: 1rem; color: #E6ECF3; display: flex; align-items: center; gap: .75rem; }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p span[data-testid="stIconMaterial"],
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p span:first-child {{ font-size: 1.3rem; width: 1.4rem; color: #9FB3CC; }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{ background: rgba(255, 255, 255, .08); }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {{ background: {CRIMSON};
  box-shadow: 0 2px 10px rgba(195, 13, 53, .45); }}
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p,
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p span {{ color: #FFFFFF !important; font-weight: 700; }}
/* navy carries the layout, crimson is the accent */
[data-testid="stMetric"] {{ background: #F4F7FB; border: 1px solid {GRID}; border-left: 4px solid {NAVY};
  border-radius: 12px; padding: .8rem 1rem; }}
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


def page_header(title, subtitle_html):
    """Navy banner with a crimson accent bar - the page title block."""
    return (f"<div style='background:linear-gradient(120deg,{NAVY} 0%,#0B2F5C 70%,#123E73 100%);border-radius:16px;"
            f"padding:1.3rem 1.6rem 1.2rem;margin:0 0 1.2rem;box-shadow:0 4px 18px rgba(4,29,59,.18)'>"
            f"<div style='width:44px;height:4px;background:{CRIMSON};border-radius:2px;margin-bottom:.75rem'></div>"
            f"<div style='font-family:Archivo,sans-serif;font-size:2rem;font-weight:700;color:#fff;line-height:1.15'>{title}</div>"
            f"<div style='color:#C9D6E6;font-size:.82rem;margin-top:.5rem;line-height:1.5'>{subtitle_html}</div></div>")


# loading overlay: the mark's three pieces drift apart and snap back together while the app is busy. Shown only while
# Streamlit's own "running" indicator (stStatusWidget) is on the page, and only after 0.6 s, so quick reruns don't flash.
_LOADER_CSS = """<style>
.kre-loader { position: fixed; inset: 0; z-index: 999990; display: none; flex-direction: column; align-items: center;
  justify-content: center; gap: 18px; background: rgba(255, 255, 255, .78); backdrop-filter: blur(3px); opacity: 0; }
body:has([data-testid="stStatusWidget"]) .kre-loader { display: flex; animation: kre-in .3s ease .6s forwards; }
@keyframes kre-in { to { opacity: 1; } }
body:has([data-testid="stChatInput"]) .kre-loader { display: none !important; }   /* assistant: has its own "Thinking..." */
.kre-mark { position: relative; width: 92px; height: 112px; animation: kre-breathe 1.8s ease-in-out infinite; }
.kre-mark img { position: absolute; inset: 0; width: 100%; height: 100%; }
.kre-mark .kre-grey { animation: kre-grey 1.8s cubic-bezier(.65, 0, .35, 1) infinite; }
.kre-mark .kre-red  { animation: kre-red  1.8s cubic-bezier(.65, 0, .35, 1) infinite; }
.kre-mark .kre-navy { animation: kre-navy 1.8s cubic-bezier(.65, 0, .35, 1) infinite; }
@keyframes kre-grey  { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(-20px, -18px) rotate(-8deg); } }
@keyframes kre-red   { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(20px, -10px) rotate(7deg); } }
@keyframes kre-navy  { 0%, 12%, 88%, 100% { transform: none; } 50% { transform: translate(-8px, 22px) rotate(-4deg); } }
@keyframes kre-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(.92); } }
.kre-loader-text { font-family: Archivo, Roboto, sans-serif; font-weight: 600; font-size: 1rem; color: __NAVY__;
  letter-spacing: .01em; }
.kre-loader-sub { font-size: .8rem; color: #5B6470; margin-top: -10px; }
@media (prefers-reduced-motion: reduce) {
  .kre-mark, .kre-mark img { animation: none !important; }
  .kre-mark { animation: kre-pulse 1.6s ease-in-out infinite !important; }
  @keyframes kre-pulse { 50% { opacity: .45; } }
}
</style>""".replace("__NAVY__", NAVY)


def loader(text="Running the flood model", sub="hazard, damage and loss for 600 buildings"):
    """The overlay markup (CSS + the three stacked pieces). Put it on the page once per run."""
    return (_LOADER_CSS + "<div class='kre-loader' role='status' aria-live='polite'><div class='kre-mark'>"
            f"<img class='kre-grey' src='{_uri('mark_grey.png')}' alt=''>"
            f"<img class='kre-navy' src='{_uri('mark_navy.png')}' alt=''>"
            f"<img class='kre-red' src='{_uri('mark_red.png')}' alt=''></div>"
            f"<div class='kre-loader-text'>{text}…</div><div class='kre-loader-sub'>{sub}</div></div>")



# headline number cards: a responsive grid (cards wrap to a new row instead of squeezing), label + value + sub-line all
# inside the card, value sized to fit. One card may be marked key (crimson edge) - the figure the page is about.
_KPI_CSS = """<style>
.kre-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 14px; margin: .2rem 0 1.4rem; }
.kre-kpi { background: #F4F7FB; border: 1px solid __GRID__; border-left: 4px solid __NAVY__; border-radius: 12px;
  padding: 14px 16px 13px; min-width: 0; }
.kre-kpi.key { border-left-color: __CRIMSON__; background: #FFF7F9; }
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
