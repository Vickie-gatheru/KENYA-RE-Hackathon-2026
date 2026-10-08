"""Visual identity: Kenya Re's public website palette and fonts (kenyare.co.ke theme CSS: --knre-global-color #C30D35,
--knre-secondary-color #041D3B, --knre-light-color #ECF0F4; headings Archivo, body Roboto). Colours only - no logo,
because this is a hackathon prototype, not a Kenya Re product.

Chart colours were checked with the dataviz palette validator (lightness band, chroma, colour-blind separation):
brand navy #041D3B is too dark/grey to be a data series, so it is used for text, headings and the one emphasised
line; series use SERIES in fixed order. Status colours (risk bands, claim verdicts, hotspot hit/miss) are kept
separate on purpose: green / amber / red must keep meaning good / warning / bad.
"""
import plotly.graph_objects as go
import plotly.io as pio

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
[data-testid="stExpander"] summary {{ background: #F4F7FB; }}
[data-testid="stExpander"] summary:hover p {{ color: {NAVY}; }}
/* chat: user turns as right-aligned bubbles, assistant turns plain (ChatGPT / Claude style) */
[data-testid="stChatMessage"] {{ background: transparent; max-width: 820px; margin: 0 auto; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{ justify-content: flex-end !important; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] {{ display: none; }}
[aria-label="Chat message from user"] {{ background: {LIGHT}; border-radius: 18px; padding: .55rem 1rem;
  flex: 0 1 auto !important; width: fit-content; max-width: 75%; margin-left: auto !important; margin-right: 0 !important; }}
[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid^="stChatMessageAvatar"] {{
  background: {CRIMSON}; color: #fff; border: none; }}
[data-testid="stChatInput"] {{ max-width: 820px; margin: 0 auto; }}
</style>"""

_DROP = ("<svg width='26' height='26' viewBox='0 0 24 24' fill='#fff' aria-hidden='true'>"
         "<path d='M12 2.7c-.3 0-.6.1-.8.4C9.5 5.3 5.5 10.4 5.5 14a6.5 6.5 0 0 0 13 0c0-3.6-4-8.7-5.7-10.9a1 1 0 0 0-.8-.4z'/></svg>")
SIDEBAR_BRAND = (f"<div style='display:flex;align-items:center;gap:.8rem'>"
                 f"<div style='background:{CRIMSON};border-radius:12px;width:46px;height:46px;display:flex;"
                 f"align-items:center;justify-content:center;flex:none;box-shadow:0 2px 8px rgba(195,13,53,.45)'>{_DROP}</div>"
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
