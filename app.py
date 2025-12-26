from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
import re

# nltk.download("punkt")
# nltk.download("stopwords")

app = Flask(__name__)
CORS(app)

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ---------------- SKILLS CONFIG ----------------

CANONICAL_SKILLS = {
    "lead generation",
    "client relationship management",
    "negotiation",
    "crm",
    "ms excel",
    "seo",
    "search engine marketing",
    "social media marketing",
    "google analytics",
    "content creation",
    "html",
    "css",
    "javascript",
    "python",
}

SOFT_SKILLS = {
    "communication",
    "teamwork",
    "leadership",
    "problem solving",
    "time management",
    "critical thinking",
    "target oriented",
}

DISPLAY_LABELS = {
    "crm": "Client Relationship Management",
    "lead generation": "Lead Generation",
    "client relationship management": "Client Relationship Management",
    "negotiation": "Negotiation",
    "communication": "Communication",
    "target oriented": "Target Oriented",
}

TEXT_WEIGHT = 0.6
SKILL_WEIGHT = 0.4

# ---------------- HELPERS ----------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_phrases(text):
    tokens = normalize_text(text).split()
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    phrases = []
    for n in (1, 2, 3):
        for gram in ngrams(tokens, n):
            phrase = " ".join(gram)
            if phrase in CANONICAL_SKILLS:
                phrases.append(phrase)
    return list(set(phrases))

def fuzzy_match(skill, text):
    return fuzz.partial_ratio(skill, text) >= 75

def semantic_match(skills, resume_text):
    found, missing = [], []
    resume_clean = normalize_text(resume_text)
    resume_emb = get_model().encode(resume_clean, convert_to_tensor=True)

    for skill in skills:
        if skill in resume_clean or fuzzy_match(skill, resume_clean):
            found.append(skill)
        else:
            skill_emb = get_model().encode(skill, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(skill_emb, resume_emb).item()
            if sim >= 0.6:
                found.append(skill)
            else:
                missing.append(skill)

    return found, missing

def detect_soft_skills(jd_text, resume_text):
    found, missing = [], []
    jd_clean = normalize_text(jd_text)
    resume_clean = normalize_text(resume_text)

    for skill in SOFT_SKILLS:
        if skill in jd_clean:
            if skill in resume_clean or fuzzy_match(skill, resume_clean):
                found.append(skill)
            else:
                missing.append(skill)
    return found, missing

def highlight_jd_text(jd_text, skills):
    highlighted = jd_text
    for skill in sorted(skills, key=len, reverse=True):
        pattern = re.compile(re.escape(skill), re.IGNORECASE)
        highlighted = pattern.sub(
            r'<span class="highlight">\g<0></span>', highlighted
        )
    return highlighted

def calculate_skill_strength(resume_text, skills):
    report = {}
    for skill in skills:
        count = len(re.findall(rf"\b{re.escape(skill)}\b", resume_text, re.I))
        if count >= 2:
            report[skill] = "Strong"
        elif count == 1:
            report[skill] = "Moderate"
        else:
            report[skill] = "Missing"
    return report

def calculate_keyword_density(resume_text, skills):
    total_words = len(resume_text.split())
    data = {}
    for skill in skills:
        count = len(re.findall(rf"\b{re.escape(skill)}\b", resume_text, re.I))
        density = round((count / total_words) * 100, 2) if total_words else 0
        status = "Well Optimized" if count >= 3 else "Needs Improvement" if count else "Missing"
        data[skill] = {"count": count, "density": density, "status": status}
    return data

def prettify(skills):
    return sorted({DISPLAY_LABELS.get(s, s.title()) for s in skills})

# ---------------- CORE ANALYSIS ----------------

def analyze_logic(resume_text, jd_text):
    emb_r = get_model().encode(normalize_text(resume_text), convert_to_tensor=True)
    emb_j = get_model().encode(normalize_text(jd_text), convert_to_tensor=True)

    text_score = round(util.pytorch_cos_sim(emb_r, emb_j).item() * 100, 2)

    jd_skills = extract_phrases(jd_text)
    found_hard, missing_hard = semantic_match(jd_skills, resume_text)
    found_soft, missing_soft = detect_soft_skills(jd_text, resume_text)

    found = prettify(found_hard + found_soft)
    missing = prettify(missing_hard + missing_soft)

    skill_score = round(
        (len(found) / (len(found) + len(missing))) * 100, 2
    ) if (found or missing) else 0

    hybrid = round(text_score * TEXT_WEIGHT + skill_score * SKILL_WEIGHT, 2)

    skill_strength = calculate_skill_strength(resume_text, found + missing)
    keyword_density = calculate_keyword_density(resume_text, found + missing)

    improved_score = min(hybrid + len(missing) * 3, 100)

    return {
        "match_score": text_score,
        "skill_match_score": skill_score,
        "hybrid_match_score": hybrid,
        "found_skills": found,
        "missing_skills": missing,
        "highlighted_jd": highlight_jd_text(jd_text, found),
        "skill_strength": skill_strength,
        "keyword_density": keyword_density,
        "improved_score_prediction": improved_score,
        "suggestions": (
            "Consider adding or emphasizing: " + ", ".join(missing)
            if missing else "Excellent match 🎉"
        )
    }


# ---------------- ROUTES ----------------

@app.route("/", methods=["GET"])
def home():
    return "Flask is running OK"


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    data = request.get_json(force=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    return jsonify(
        analyze_logic(
            data.get("resume", ""),
            data.get("jd", "")
        )
    )


@app.route("/uploadResume", methods=["POST"])
def upload_resume():
    jd_text = request.form.get("jdText", "")
    resume_text = request.files["resumeFile"].read().decode("utf-8", errors="ignore")
    return jsonify(analyze_logic(resume_text, jd_text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

