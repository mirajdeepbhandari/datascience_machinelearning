

# pip install streamlit transformers peft torch pandas accelerate

import html
import json
import os
import re
import ssl

import requests
import urllib3


# ---------------------------------------------------------------------------
# SSL patch (must run before any HF download)
# ---------------------------------------------------------------------------
def disable_ssl_verification():
    """Disable SSL certificate verification globally for requests + httpx."""
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Patch requests
    original_request = requests.Session.request

    def request_without_ssl(self, *args, **kwargs):
        kwargs["verify"] = False
        return original_request(self, *args, **kwargs)

    requests.Session.request = request_without_ssl

    # Patch httpx (used by huggingface_hub)
    try:
        import httpx

        _orig_client_init = httpx.Client.__init__
        _orig_async_init = httpx.AsyncClient.__init__

        def _client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_client_init(self, *args, **kwargs)

        def _async_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _orig_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _client_init
        httpx.AsyncClient.__init__ = _async_init
    except ImportError:
        pass

    # Nuke default SSL context
    ssl._create_default_https_context = ssl._create_unverified_context


disable_ssl_verification()
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "mirajbhandari/Entity_Extcation_Quen"

SYSTEM_PROMPT = (
    "You are an NER model. Extract named entities from the sentence and "
    'return ONLY a JSON list of objects with keys "text" and "type". '
    "Allowed types: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, "
    "MONEY, TIME, WORK_OF_ART, LANGUAGE, NORP, FAC, GPE."
)

# Distinct color per entity type
ENTITY_COLORS = {
    "PERSON": "#FFADAD",
    "ORGANIZATION": "#FFD6A5",
    "LOCATION": "#FDFFB6",
    "DATE": "#CAFFBF",
    "EVENT": "#9BF6FF",
    "PRODUCT": "#A0C4FF",
    "MONEY": "#BDB2FF",
    "TIME": "#FFC6FF",
    "WORK_OF_ART": "#FFB4A2",
    "LANGUAGE": "#B5EAD7",
    "NORP": "#E2F0CB",
    "FAC": "#C7CEEA",
    "GPE": "#F7D6E0",
}
DEFAULT_COLOR = "#E0E0E0"


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.to(device).eval()
    return tokenizer, model, device


def build_messages(sentence):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]


def predict(sentence, tokenizer, model, device, max_new_tokens=256):
    import torch

    prompt = tokenizer.apply_chat_template(
        build_messages(sentence), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    try:
        data = json.loads(gen)
    except Exception:
        m = re.search(r"\[.*\]", gen, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else []
        except Exception:
            data = []

    if not isinstance(data, list):
        return []

    cleaned = []
    for e in data:
        if isinstance(e, dict) and "text" in e and "type" in e:
            cleaned.append(
                {
                    "text": str(e["text"]).strip(),
                    "type": str(e["type"]).strip().upper(),
                }
            )
    return cleaned


# ---------------------------------------------------------------------------
# Highlight rendering
# ---------------------------------------------------------------------------
def highlight_text(sentence, entities):
    """Return HTML with entity spans highlighted (case-insensitive, longest-first)."""
    if not entities:
        return f"<div class='ner-text'>{html.escape(sentence)}</div>"

    ents_sorted = sorted(entities, key=lambda e: len(e["text"]), reverse=True)

    spans = []
    used = [False] * len(sentence)
    for e in ents_sorted:
        if not e["text"]:
            continue
        for m in re.finditer(re.escape(e["text"]), sentence, re.IGNORECASE):
            s, en = m.start(), m.end()
            if any(used[s:en]):
                continue
            spans.append((s, en, e["type"]))
            for i in range(s, en):
                used[i] = True

    spans.sort(key=lambda x: x[0])

    out = []
    cursor = 0
    for s, en, typ in spans:
        if cursor < s:
            out.append(html.escape(sentence[cursor:s]))
        color = ENTITY_COLORS.get(typ, DEFAULT_COLOR)
        out.append(
            f'<span class="ent" style="background:{color};">'
            f"{html.escape(sentence[s:en])}"
            f'<span class="ent-label">{html.escape(typ)}</span>'
            f"</span>"
        )
        cursor = en
    if cursor < len(sentence):
        out.append(html.escape(sentence[cursor:]))

    return f"<div class='ner-text'>{''.join(out)}</div>"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NER Inference", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .ner-text {
        font-size: 1.15rem;
        line-height: 2.4;
        padding: 1.2rem 1.4rem;
        background: #FAFAFA;
        border: 1px solid #E5E5E5;
        border-radius: 12px;
        color: #111;
    }
    .ent {
        padding: 0.15em 0.45em;
        margin: 0 0.15em;
        border-radius: 6px;
        color: #111;
        font-weight: 500;
    }
    .ent-label {
        font-size: 0.65em;
        font-weight: 700;
        margin-left: 0.4em;
        padding: 0.1em 0.4em;
        background: rgba(0,0,0,0.65);
        color: #fff;
        border-radius: 4px;
        vertical-align: middle;
        letter-spacing: 0.05em;
    }
    .legend-chip {
        display: inline-block;
        padding: 0.2em 0.7em;
        margin: 0.15em;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Named Entity Recognition")

with st.sidebar:
    st.header("⚙️ Settings")
    max_new = st.slider("Max new tokens", 64, 512, 256, 32)
    st.divider()
    st.subheader("🎨 Entity Legend")
    legend_html = "".join(
        f'<span class="legend-chip" style="background:{c};">{t}</span>'
        for t, c in ENTITY_COLORS.items()
    )
    st.markdown(legend_html, unsafe_allow_html=True)

tokenizer, model, device = load_model()
st.caption(f"Device: `{device}`")

default_text = "Elon Musk visited Nepal last Monday to open a new Tesla factory."
text = st.text_area("Enter a sentence", value=default_text, height=120)

col1, _ = st.columns([1, 5])
with col1:
    run = st.button("✨ Extract", type="primary", use_container_width=True)

if run and text.strip():
    with st.spinner("Extracting entities..."):
        entities = predict(text.strip(), tokenizer, model, device, max_new)

    st.subheader("Highlighted")
    st.markdown(highlight_text(text, entities), unsafe_allow_html=True)

    st.subheader("Extracted Entities")
    if entities:
        df = pd.DataFrame(entities)
        df.index = df.index + 1
        df.index.name = "#"
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(entities))
        c2.metric("Unique", df["text"].str.lower().nunique())
        c3.metric("Types", df["type"].nunique())

        with st.expander("Raw JSON"):
            st.json(entities)
    else:
        st.info("No entities detected.")
elif run:
    st.warning("Please enter some text.")
