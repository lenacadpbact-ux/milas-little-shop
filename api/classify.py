from http.server import BaseHTTPRequestHandler
import json
import os
import io
import re
import urllib.request
import urllib.error
import joblib
import numpy as np
import scipy.sparse as sp

BASE_DIR = os.path.dirname(__file__)

# --------------------------------------------------
# Constants
# --------------------------------------------------
VALID_UIDS = ['[EDU_01]', '[TOP_02]', '[EMO_03]', '[BRD_04]', '[GEN_05]']

CATEGORY_LABELS = {
    '[EDU_01]': 'Grade-Specific',
    '[TOP_02]': 'Topic-Specific',
    '[EMO_03]': 'Emotive / Resilience',
    '[BRD_04]': 'Brand-Specific',
    '[GEN_05]': 'General',
}

# Used only if the label-encoder file is missing: LabelEncoder sorts labels
# alphabetically, so 0..4 map to this order (verify against your notebook)
FALLBACK_CLASSES = ['BRD_04', 'EDU_01', 'EMO_03', 'GEN_05', 'TOP_02']

BLOB_URL = "https://itf8egqp4iag2pr5.private.blob.vercel-storage.com/celm_rf_model.joblib"

# --------------------------------------------------
# AWS Bedrock settings (set CELM_BEDROCK_KEY in Vercel; never put it in code)
# --------------------------------------------------
_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1").strip()
if re.fullmatch(r"[a-z]{2}-[a-z]+-\d", _region):
    AWS_REGION = _region
else:
    print("Warning: AWS_DEFAULT_REGION is not a valid region; using us-east-1")
    AWS_REGION = "us-east-1"

BEDROCK_KEY = os.environ.get("CELM_BEDROCK_KEY", "").strip()
print("CELM Bedrock key present:", bool(BEDROCK_KEY))

BEDROCK_MODEL_ID = "us.amazon.nova-lite-v1:0"
BEDROCK_URL = (
    f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/model/"
    f"{BEDROCK_MODEL_ID}/converse"
)

# --------------------------------------------------
# PIPELINE A: signal dictionaries (from the Experiment 4 script)
# --------------------------------------------------
EDU_SIGNALS = {
    'USA': ['1st grade', 'first grade', 'grade 1', '2nd grade', 'grade 2', 'kindergarten', 'k-4'],
    'UK':  ['year 1', 'year 2', 'ks1', 'ks2', 'key stage 1', 'key stage 2', 'reception'],
    'NZ':  ['year 1', 'year 2', 'year 3', 'new zealand curriculum'],
    'AU':  ['year 1', 'year 2', 'foundation year', 'prep', 'australian curriculum'],
    'JP':  ['小学1年生', '1年生', '小学２年生', '小学一年生'],
    'CN':  ['一年级', '二年级', '小学一年级', '小学二年级'],
    'PH':  ['baitang 1', 'baitang 2', 'baitang 3', 'grade 1', 'grade 2', 'kinder', 'k-12', 'deped', 'unang baitang'],
}
TOPIC_SIGNALS = ['geography', 'world', 'stem', 'science', 'history', 'maths', 'math',
                 'maps', 'landmarks', 'countries', 'nature', 'environment', 'technology',
                 'engineering', 'araling panlipunan', 'agham', 'matematika', 'heograpiya',
                 'kasaysayan', 'mapeh', '地理', '科学', '数学', '理科', '社会']
EMO_SIGNALS = ['belonging', 'resilience', 'emotions', 'friendship', 'feelings', 'home',
               'brave', 'courage', 'identity', 'inclusion', 'diversity', 'moving',
               'new beginnings', 'bayanihan', 'pagmamahal', 'pagkakaibigan', 'pamilya',
               'malasakit', 'tapang']
BRAND_SIGNALS = ['mila', 'morphle', 'julia donaldson', 'roald dahl', 'jeannie baker',
                 'bluey', 'dog man', 'adarna', 'lola basyang', 'lampara',
                 'anvil publishing', 'tahanan books', 'rene villanueva']

CELM_PROMPT = (
    "You are CELM — assign exactly one Universal ID to the given book title.\n\n"
    "[EDU_01] Grade-Specific — explicitly mentions a curriculum grade or year "
    "group: Year 1, Year 2, KS1, KS2, 1st Grade, 2nd Grade, kindergarten, "
    "K-4, Baitang 1, Baitang 2, DepEd, K-12. Age ranges alone without a grade "
    "term do not qualify.\n\n"
    "[TOP_02] Topic-Specific — focused on a curriculum subject or theme: "
    "geography, science, STEM, history, maths, world, landmarks, maps, "
    "countries, culture, nature, Araling Panlipunan, Agham, Matematika. "
    "May include age ranges like Ages 6-8 alongside a topic.\n\n"
    "[EMO_03] Emotive/Resilience — social emotional themes: belonging, "
    "resilience, emotions, friendship, feelings, home, brave, courage, "
    "identity, inclusion, diversity, moving, new beginnings, Bayanihan, "
    "Pagmamahal, Pagkakaibigan.\n\n"
    "[BRD_04] Brand-Specific — known children's book brands or authors: "
    "Mila, Morphle, Julia Donaldson, Roald Dahl, Jeannie Baker, Bluey, "
    "Adarna House, Lola Basyang, Lampara Books, Anvil Publishing.\n\n"
    "[GEN_05] General — general children's content not clearly fitting the "
    "above: picture book, bestselling, award winning, illustrated, stories, "
    "adventure, pambata, kwento, aklat.\n\n"
    "Return ONLY the Universal ID including square brackets. Example: [EDU_01]"
)


def calculate_signal_strength(title, country='USA'):
    score = 1
    t = str(title).lower()
    if any(k in t for k in EDU_SIGNALS.get(country, [])):
        score += 5
    if any(k in t for k in TOPIC_SIGNALS):
        score += 3
    if any(k in t for k in EMO_SIGNALS):
        score += 3
    if any(k in t for k in BRAND_SIGNALS):
        score += 4
    return max(1, min(10, score))


def predict_pipeline_a(title, country='USA'):
    t = str(title).lower()
    if any(k in t for k in EDU_SIGNALS.get(country, [])):
        return '[EDU_01]'
    if any(k in t for k in BRAND_SIGNALS):
        return '[BRD_04]'
    if any(k in t for k in EMO_SIGNALS):
        return '[EMO_03]'
    if any(k in t for k in TOPIC_SIGNALS):
        return '[TOP_02]'
    return '[GEN_05]'


# --------------------------------------------------
# Model loading: done on the first request, not at import time, so a loading
# problem gives a readable error instead of a bare 500 page
# --------------------------------------------------
_models = {}


def get_models():
    if _models:
        return _models

    _models["vectorizer"] = joblib.load(
        os.path.join(BASE_DIR, "../model/celm_tfidf_vectorizer.joblib"))

    encoder_path = os.path.join(BASE_DIR, "../model/celm_label_encoder.joblib")
    _models["encoder"] = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

    token = os.environ["BLOB_READ_WRITE_TOKEN"]
    req = urllib.request.Request(BLOB_URL, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rf_bytes = resp.read()
    _models["rf"] = joblib.load(io.BytesIO(rf_bytes))
    del rf_bytes

    print("RF classes:", list(_models["rf"].classes_),
          "| encoder loaded:", _models["encoder"] is not None)
    return _models


# --------------------------------------------------
# Pipeline B: decode the forest's output into bracketed UIDs
# --------------------------------------------------
def to_bracketed(label):
    s = str(label).strip()
    return s if s.startswith('[') else '[' + s + ']'


def decode_prediction(pred, encoder):
    s = str(pred).strip()
    if s.startswith('['):
        return s
    if s.isdigit():
        idx = int(s)
        if encoder is not None:
            return to_bracketed(encoder.inverse_transform([idx])[0])
        if idx < len(FALLBACK_CLASSES):
            return to_bracketed(FALLBACK_CLASSES[idx])
    return to_bracketed(s)


def predict_pipeline_b(title, signal_strength):
    m = get_models()
    tfidf_vec = m["vectorizer"].transform([title])
    ss_norm = np.array([[signal_strength / 10.0]])
    full_vec = sp.hstack([tfidf_vec, ss_norm])
    return decode_prediction(m["rf"].predict(full_vec)[0], m["encoder"])


# --------------------------------------------------
# LLM tie-breaker (direct Bedrock call with an API key; returns answer + status)
# --------------------------------------------------
def predict_llm_tiebreaker(title, uid_a, uid_b, signal_strength):
    if not BEDROCK_KEY:
        print("LLM skipped: CELM_BEDROCK_KEY is not set")
        return uid_b, "not_configured"

    tie_break_prompt = CELM_PROMPT + (
        f"\n\nTWO CLASSIFIERS DISAGREE ON THIS TITLE.\n"
        f"Rule-based evidence system proposed: {uid_a} (Signal Strength: {signal_strength}/10)\n"
        f"Statistical text classifier proposed: {uid_b}\n"
        f"Using the definitions above and the title itself, decide which of these "
        f"two candidates is correct, or select a different Universal ID if neither fits.\n"
        f"Return ONLY the Universal ID."
    )
    body = json.dumps({
        "system": [{"text": tie_break_prompt}],
        "messages": [{"role": "user", "content": [{"text": f'Title: "{title}"'}]}],
        "inferenceConfig": {"maxTokens": 10, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        BEDROCK_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {BEDROCK_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            out = json.loads(resp.read())
        text = out["output"]["message"]["content"][0]["text"]
        uid = to_bracketed(text.strip().split("\n")[0])
        if uid in VALID_UIDS:
            return uid, "ok"
        return uid_b, "invalid_response"
    except urllib.error.HTTPError as e:
        print(f"LLM HTTP error {e.code}: {e.read()[:200]}")
        return uid_b, "error"
    except Exception as e:
        print(f"LLM error: {str(e)[:200]}")
        return uid_b, "error"


# --------------------------------------------------
# Full classification
# --------------------------------------------------
def classify_title(title, country='USA'):
    signal_strength = calculate_signal_strength(title, country)
    a = predict_pipeline_a(title, country)
    b = predict_pipeline_b(title, signal_strength)

    # Guard: stop if Pipeline B returned something that isn't a valid UID
    if b not in VALID_UIDS:
        raise ValueError(f"Unexpected Pipeline B output: {b}")

    agreed = (a == b)

    if agreed:
        final_uid, llm_status = a, "not_needed"
        note = "Both pipelines agreed on this category."
    else:
        llm_uid, llm_status = predict_llm_tiebreaker(title, a, b, signal_strength)
        if llm_status == "ok":
            final_uid = llm_uid
            note = "The pipelines disagreed, so a language model made the final call."
        else:
            final_uid = a
            note = "The pipelines disagreed and the language model was unavailable, so the rule-based result was used."

    return {
        "title": title,
        "country": country,
        "final_uid": final_uid,
        "final_label": CATEGORY_LABELS.get(final_uid, final_uid),
        "pipeline_a_label": CATEGORY_LABELS.get(a, a),
        "pipeline_b_label": CATEGORY_LABELS.get(b, b),
        "signal_strength": signal_strength,
        "agreed": agreed,
        "llm_status": llm_status,
        "explanation": note,
    }


# --------------------------------------------------
# HTTP handler (Vercel entry point)
# --------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        query = parse_qs(urlparse(self.path).query)
        title = query.get('title', [''])[0].strip()[:200]
        country = query.get('country', ['USA'])[0].upper()
        if country not in EDU_SIGNALS:
            country = 'USA'

        if not title:
            return self._send(400, {"error": "Missing 'title' parameter"})
        try:
            result = classify_title(title, country)
            print(json.dumps(result, ensure_ascii=False))   # request log for monitoring
            self._send(200, result)
        except Exception as e:
            print(f"Classification error: {type(e).__name__}: {str(e)[:300]}")
            self._send(500, {"error": "Classification failed. Please try again."})
