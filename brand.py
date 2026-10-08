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
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .3rem .4rem; border-radius: 6px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: rgba(195, 13, 53, .25); }}
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

SIDEBAR_BRAND = (f"<div style='font-family:Archivo,sans-serif;font-weight:700;font-size:1.25rem;color:#fff;"
                 f"line-height:1.2'>Nairobi Flood<br><span style='color:{CRIMSON}'>Risk Workbench</span></div>"
                 "<div style='font-size:.75rem;opacity:.75;margin-top:.3rem'>Kenya Re AI4I Hackathon 2026 · Team A"
                 "<br>Prototype - not a Kenya Re product</div>")
