from http.server import BaseHTTPRequestHandler
import json
import os
import io
import urllib.request
import joblib
import numpy as np
import scipy.sparse as sp
import litellm

BASE_DIR = os.path.dirname(__file__)

# --------------------------------------------------
# Load models (vectorizer + encoder from GitHub, forest from private Blob)
# --------------------------------------------------
BLOB_READ_WRITE_TOKEN = os.environ["BLOB_READ_WRITE_TOKEN"]

vectorizer = joblib.load(os.path.join(BASE_DIR, "../model/celm_tfidf_vectorizer.joblib"))

ENCODER_PATH = os.path.join(BASE_DIR, "../model/celm_label_encoder.joblib")
label_encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

BLOB_URL = "https://itf8egqp4iag2pr5.private.blob.vercel-storage.com/celm_rf_model.joblib"
_req = urllib.request.Request(BLOB_URL, headers={"Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}"})
with urllib.request.urlopen(_req) as _resp:
    _rf_bytes = _resp.read()
rf_model = joblib.load(io.BytesIO(_rf_bytes))
del _rf_bytes

print("RF classes:", list(rf_model.classes_), "| encoder loaded:", label_encoder is not None)

# --------------------------------------------------
# Constants
# --------------------------------------------------
LLM_MODEL = 'bedrock/us.amazon.nova-lite-v1:0'
VALID_UIDS = ['[EDU_01]', '[TOP_02]', '[EMO_03]', '[BRD_04]', '[GEN_05]']

CATEGORY_LABELS = {
    '[EDU_01]': 'Grade-Specific',
    '[TOP_02]': 'Topic-Specific',
    '[EMO_03]': 'Emotive / Resilience',
    '[BRD_04]': 'Brand-Specific',
    '[GEN_05]': 'General',
}

# --------------------------------------------------
# PIPELINE A — signal dictionaries (from Experiment 4 script)
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
    if any(k in t for k in EDU_SIGNALS.get(country, [])): score += 5
    if any(k in t for k in TOPIC_SIGNALS): score += 3
    if any(k in t for k in EMO_SIGNALS): score += 3
    if any(k in t for k in BRAND_SIGNALS): score += 4
    return max(1, min(10, score))

def predict_pipeline_a(title, country='USA'):
    t = str(title).lower()
    if any(k in t for k in EDU_SIGNALS.get(country, [])): return '[EDU_01]'
    if any(k in t for k in BRAND_SIGNALS): return '[BRD_04]'
    if any(k in t for k in EMO_SIGNALS): return '[EMO_03]'
    if any(k in t for k in TOPIC_SIGNALS): return '[TOP_02]'
    return '[GEN_05]'

# --------------------------------------------------
# Pipeline B: decode the forest's output into bracketed UIDs
# --------------------------------------------------
def to_bracketed(label):
    s = str(label).strip()
    return s if s.startswith('[') else '[' + s + ']'

def decode_prediction(pred):
    if label_encoder is not None and isinstance(pred, (int, np.integer)):
        return to_bracketed(label_encoder.inverse_transform([pred])[0])
    return to_bracketed(pred)

def predict_pipeline_b(title, signal_strength):
    tfidf_vec = vectorizer.transform([title])
    ss_norm = np.array([[signal_strength / 10.0]])
    full_vec = sp.hstack([tfidf_vec, ss_norm])
    return decode_prediction(rf_model.predict(full_vec)[0])

# --------------------------------------------------
# LLM tie-breaker (returns the answer plus a status)
# --------------------------------------------------
def predict_llm_tiebreaker(title, uid_a, uid_b, signal_strength):
    tie_break_prompt = CELM_PROMPT + (
        f"\n\nTWO CLASSIFIERS DISAGREE ON THIS TITLE.\n"
        f"Rule-based evidence system proposed: {uid_a} (Signal Strength: {signal_strength}/10)\n"
        f"Statistical text classifier proposed: {uid_b}\n"
        f"Using the definitions above and the title itself, decide which of these "
        f"two candidates is correct, or select a different Universal ID if neither fits.\n"
        f"Return ONLY the Universal ID."
    )
    try:
        r = litellm.completion(
            model=LLM_MODEL,
            messages=[{'role': 'system', 'content': tie_break_prompt},
                      {'role': 'user', 'content': f'Title: "{title}"'}],
            max_tokens=10,
            temperature=0.0,
        )
        uid = to_bracketed(r.choices[0].message.content.strip().split('\n')[0])
        if uid in VALID_UIDS:
            return uid, "ok"
        return uid_b, "invalid_response"
    except Exception as e:
        print(f"LLM error: {str(e)[:100]}")
        return uid_b, "error"

# --------------------------------------------------
# Full classification
# --------------------------------------------------
def classify_title(title, country='USA'):
    signal_strength = calculate_signal_strength(title, country)
    a = predict_pipeline_a(title, country)
    b = predict_pipeline_b(title, signal_strength)

    # Guard: stop here if Pipeline B returned something that isn't a valid UID
    if b not in VALID_UIDS:
        raise ValueError(f"Unexpected Pipeline B output: {b}")

    agreed = (a == b)

    if agreed:
        final_uid, llm_status = a, "not_needed"
        note = "Both pipelines agreed on this category."
    else:
        final_uid, llm_status = predict_llm_tiebreaker(title, a, b, signal_strength)
        if llm_status == "ok":
            note = "The pipelines disagreed, so a language model made the final call."
        else:
            note = "The pipelines disagreed and the language model was unavailable, so the data-driven result was used."

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
# HTTP handler
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
            print(f"Classification error: {str(e)[:200]}")
            self._send(500, {"error": "Classification failed. Please try again."})
