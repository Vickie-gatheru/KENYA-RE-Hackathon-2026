"""One place for LLM calls. Provider from env LLM_PROVIDER = groq | gemini | anthropic | openai | manual.
Model from env LLM_MODEL (defaults per provider). Keys: GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY /
OPENAI_API_KEY.  On Windows PowerShell:  $env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="gsk_..."
"""
import json, os, time
import urllib.request, urllib.error

UA = "nairobi-flood-cat/0.1"
DEFAULT_MODEL = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash",
                 "anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


class ManualMode(Exception):
    """No API provider configured."""


def provider():
    return os.environ.get("LLM_PROVIDER", "manual")


def configured():
    p = provider()
    if p == "test":
        return True
    key = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "openai": "OPENAI_API_KEY"}.get(p)
    return bool(key and os.environ.get(key))


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **headers})
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            if e.code == 429 and attempt < 5:            # free-tier rate limit: wait and retry
                wait = float(e.headers.get("retry-after") or 20)
                print(f"  rate limited, waiting {wait:.0f}s"); time.sleep(wait); continue
            if e.code == 404 and "model" in msg and provider() == "groq":
                msg += "\n" + _groq_models_hint()
            raise RuntimeError(f"{provider()} error {e.code}: {msg}") from None


def _groq_models_hint():
    try:
        req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers={
            "User-Agent": UA, "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        ids = sorted(m["id"] for m in json.load(urllib.request.urlopen(req, timeout=30))["data"])
        return "Models available to your key:\n  " + "\n  ".join(ids) + '\nSet one with  $env:LLM_MODEL="<id>"'
    except Exception:
        return ""


def complete(prompt, json_mode=True):
    p = provider()
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL.get(p, ""))
    if p == "groq":
        body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = _post("https://api.groq.com/openai/v1/chat/completions", body,
                  {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        return r["choices"][0]["message"]["content"]
    if p == "gemini":
        cfg = {"temperature": 0, **({"responseMimeType": "application/json"} if json_mode else {})}
        r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                  f"?key={os.environ['GEMINI_API_KEY']}",
                  {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}, {})
        return r["candidates"][0]["content"]["parts"][0]["text"]
    if p == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(model=model, max_tokens=4000,
                                                  messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    if p == "openai":
        from openai import OpenAI
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **kw)
        return r.choices[0].message.content
    if p == "test":   # OFFLINE TEST STUB - scripted, not a language model; used only to exercise the UI and tests
        if "Your JSON reply" in prompt:
            if "returned:" not in prompt:
                return '{"tool": "portfolio_summary", "args": {}}'
            got = json.loads(prompt.split("returned:\n", 1)[1].split("\n\nYour JSON reply")[0])
            return json.dumps({"answer": f"[TEST STUB] AAL is {got.get('technical_premium_AAL')} and the 1-in-100 "
                                         f"loss is {got.get('ep_curve', {}).get('1-in-100')}; 1-in-250 is KES 777 m."})
        return '{"risks": []}' if json_mode else "[TEST STUB memo]"
    raise ManualMode()


def parse_json(text):
    i, j = text.find("{"), text.rfind("}")
    return json.loads(text[i:j + 1])
