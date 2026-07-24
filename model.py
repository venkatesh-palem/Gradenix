"""
model.py – Gradenix prediction layer
======================================
Prediction logic mirrors the notebook exactly:
  1. Hard override  : Attendance < 35  → At Risk (always)
  2. Composite score: 50% CGPA + 25% Attendance + 15% StudyHours + 10% (inverted SocialMedia)
     mapped to labels using the saved quantile thresholds from score_thresholds.json
  3. ML model       : GradientBoostingClassifier trained on composite-score labels
     (gives confidence % via predict_proba)

All three layers agree — the hard override and score are deterministic,
the ML model was trained on the same score-derived labels.
"""
import os, json
import numpy as np
import pandas as pd
import joblib

_DIR   = os.path.dirname(__file__)
_model = None
_le    = None
_thresh = None

FEATURE_COLS = [
    "Attendance", "StudyHours", "PreviousSGPA", "SkillDevelopmentHours",
    "SocialMediaHours", "EnglishProficiency", "Scholarship", "CoCurricular",
    "HealthIssues", "CurrentSemester", "CurrentCGPA",
]
LABEL_MAP = {0: "At Risk", 1: "Average", 2: "Good", 3: "Excellent"}


def _load():
    global _model, _le, _thresh
    _model  = joblib.load(os.path.join(_DIR, "rf_model.pkl"))
    _le     = joblib.load(os.path.join(_DIR, "label_encoder.pkl"))
    _thresh = json.load(open(os.path.join(_DIR, "score_thresholds.json")))


def _composite_label(cgpa, att, sh, sm, thresh):
    """Replicate the notebook's composite-score label assignment."""
    score = (
        (cgpa / 10.0)        * 0.50 +
        (att  / 100.0)       * 0.25 +
        (sh   / 13.0)        * 0.15 +
        (1.0 - sm / 20.0)    * 0.10
    )
    score = max(0.0, min(1.0, score))
    if   score >= thresh["Q_GOOD"]:    return "Excellent"
    elif score >= thresh["Q_AVERAGE"]: return "Good"
    elif score >= thresh["Q_ATRISK"]:  return "Average"
    else:                              return "At Risk"


def predict(features: dict):
    """
    Returns (label: str, confidence: float).
    features keys: same as FEATURE_COLS (already encoded — EnglishProficiency 0/1/2, booleans 0/1).
    """
    global _model, _le, _thresh
    if _model is None:
        _load()

    att  = float(features.get("Attendance", 100))
    cgpa = float(features.get("CurrentCGPA", 0))
    sh   = float(features.get("StudyHours", 0))
    sm   = float(features.get("SocialMediaHours", 0))

    # Hard override – matches notebook Cell 20
    if att < 35:
        return "At Risk", 100.0

    # ML prediction for confidence
    row   = pd.DataFrame([[features.get(f, 0) for f in FEATURE_COLS]], columns=FEATURE_COLS)
    pred  = _model.predict(row)[0]
    proba = _model.predict_proba(row)[0]
    label = LABEL_MAP[int(pred)]
    conf  = round(float(max(proba)) * 100, 1)

    # Composite score cross-check (deterministic, same training signal)
    score_label = _composite_label(cgpa, att, sh, sm, _thresh)
    # If composite and ML agree, return as-is; if not, composite wins (it's the ground truth)
    if score_label != label:
        label = score_label
        conf  = round(max(proba) * 100 * 0.9, 1)   # slight confidence penalty for disagreement

    return label, conf


def get_tips(features: dict) -> list[str]:
    att       = float(features.get("Attendance", 100))
    cgpa      = float(features.get("CurrentCGPA", 0))
    sh        = float(features.get("StudyHours", 0))
    sm        = float(features.get("SocialMediaHours", 0))
    prev_sgpa = float(features.get("PreviousSGPA", cgpa))
    skill_hrs = float(features.get("SkillDevelopmentHours", 0))
    co_curr   = int(features.get("CoCurricular", 0))
    health    = int(features.get("HealthIssues", 0))

    tips = []

    # Attendance
    if att < 35:
        tips.append("🚨 Attendance critically low (<35%) — this alone triggers At Risk. Attend every class immediately.")
    elif att < 60:
        tips.append("⚠️ Attendance below 60% — exam eligibility is at risk. Make showing up your top priority.")
    elif att < 75:
        tips.append("🏫 Attendance below 75%. Aim for at least 85% to stay safe and retain lectures properly.")

    # CGPA
    if cgpa < 5.0:
        tips.append("🎯 CGPA critically low. Seek academic counselling and tutoring now — every grade point matters.")
    elif cgpa < 6.87:
        tips.append("🎯 CGPA is in the At Risk zone. Focus on fundamentals and ask faculty for extra guidance.")
    elif cgpa < 8.12:
        tips.append("📈 Average band. Consistent daily effort and fewer distractions can push you into Good tier.")
    elif cgpa < 9.37:
        tips.append("🌟 Good standing! Push study quality and cut low-value screen time to reach Excellent.")

    # Study hours
    if sh < 1:
        tips.append("📚 Studying less than 1 hr/day is unsustainable. Start with a fixed 2-hour daily block.")
    elif sh < 2:
        tips.append("📚 Study time very low. Aim for at least 3 focused hours/day to see real improvement.")
    elif sh < 3:
        tips.append("📚 Boosting to 3+ hours daily will noticeably lift your scores this semester.")

    # Social media
    if sm > 12:
        tips.append("📵 Social media >12 hrs/day is severely damaging your performance. Set strict app limits today.")
    elif sm > 6:
        tips.append("📵 Over 6 hrs/day on social media is a major distraction. Cap it at 2–3 hours maximum.")
    elif sm > 4:
        tips.append("📵 Limit social media to under 3 hrs/day to reclaim productive study time.")

    # Previous SGPA vs Current CGPA trend
    if prev_sgpa > 0:
        diff = cgpa - prev_sgpa
        if diff < -1.0:
            tips.append(f"📉 CGPA dropped {abs(diff):.2f} points vs last semester. Identify what changed and seek early support.")
        elif diff < -0.5:
            tips.append(f"📉 Slight CGPA dip from last semester. Review study habits and tighten your schedule now.")
        elif diff > 1.0:
            tips.append(f"📈 Great improvement of {diff:.2f} points over last semester — keep this momentum going!")

    # Skill development hours
    if skill_hrs == 0:
        tips.append("🛠️ No skill development hours logged. Even 2–3 hrs/week on courses or projects boosts employability significantly.")
    elif skill_hrs < 2:
        tips.append("🛠️ Very low skill dev hours. Try dedicating at least 3 hrs/week to certifications or hands-on projects.")
    elif skill_hrs >= 10:
        tips.append("🛠️ Great skill investment! Make sure it doesn't eat into core study time — balance is key.")

    # Co-curricular
    if co_curr == 0:
        tips.append("🏅 Consider joining at least one club or activity — co-curriculars build soft skills and campus networks.")
    else:
        tips.append("🏅 Active in co-curriculars! Ensure these activities complement, not compete with, your study schedule.")

    # Health issues
    if health == 1:
        tips.append("🏥 Health challenges noted. Communicate with faculty early for accommodations and prioritise rest — recovery is part of performance.")

    # Imbalance
    if sh < 2 and sm > 6:
        tips.append("⚡ Critical imbalance: far more social media than study. Flipping this ratio alone will lift your grade.")

    if not tips:
        tips.append("✅ Excellent habits across every metric — keep this momentum and you're on track for top results!")

    return tips
