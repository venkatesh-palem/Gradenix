"""
retrain_model.py — run this once after extracting the zip to regenerate
rf_model.pkl, label_encoder.pkl, and score_thresholds.json without
FamilyIncome and CreditsCompleted.

  cd student_app
  python retrain_model.py
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

LABEL_ORDER = ['At Risk', 'Average', 'Good', 'Excellent']
SEED = 42
np.random.seed(SEED)

DIR = os.path.dirname(__file__)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DIR, 'Student_Dataset_Scaled10.csv'))
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Parse attendance (handles "70-80" range strings) ─────────────────────────
def parse_attendance(val):
    val = str(val).strip()
    if '-' in val:
        lo, hi = val.split('-')
        return (float(lo) + float(hi)) / 2
    return float(val)

df['Attendance'] = df['Attendance'].apply(parse_attendance)

# ── Encode categoricals ───────────────────────────────────────────────────────
def norm_binary(v):
    return 1 if str(v).strip().lower() in ['yes', 'y', '1'] else 0

def norm_english(v):
    return {'basic': 0, 'intermediate': 1, 'advance': 2, 'advanced': 2}.get(
        str(v).strip().lower(), 1)

df['HealthIssues']       = df['HealthIssues'].apply(norm_binary)
df['Scholarship']        = df['Scholarship'].apply(norm_binary)
df['CoCurricular']       = df['CoCurricular'].apply(norm_binary)
df['EnglishProficiency'] = df['EnglishProficiency'].apply(norm_english)

# ── Composite score → new labels (no CGPA-only cheating) ─────────────────────
df['cgpa_n'] = df['CurrentCGPA']            / 10.0
df['att_n']  = df['Attendance']              / 100.0
df['sh_n']   = df['StudyHours']              / 13.0
df['sm_n']   = 1.0 - df['SocialMediaHours'] / 20.0

df['composite_score'] = (
    df['cgpa_n'] * 0.50 +
    df['att_n']  * 0.25 +
    df['sh_n']   * 0.15 +
    df['sm_n']   * 0.10
)

Q_ATRISK  = df['composite_score'].quantile(0.20)
Q_AVERAGE = df['composite_score'].quantile(0.50)
Q_GOOD    = df['composite_score'].quantile(0.75)

def assign_label(score):
    if   score >= Q_GOOD:    return 'Excellent'
    elif score >= Q_AVERAGE: return 'Good'
    elif score >= Q_ATRISK:  return 'Average'
    else:                    return 'At Risk'

df['PerformanceCategory'] = df['composite_score'].apply(assign_label)
print("Label distribution:", df['PerformanceCategory'].value_counts().to_dict())

# ── Features — NO FamilyIncome, NO CreditsCompleted ──────────────────────────
FEATURE_COLS = [
    'Attendance', 'StudyHours', 'PreviousSGPA', 'SkillDevelopmentHours',
    'SocialMediaHours', 'EnglishProficiency', 'Scholarship', 'CoCurricular',
    'HealthIssues', 'CurrentSemester', 'CurrentCGPA',
]

le = LabelEncoder()
le.fit(LABEL_ORDER)
le.classes_ = np.array(LABEL_ORDER)

X = df[FEATURE_COLS].astype(float)
y = le.transform(df['PerformanceCategory'])

# ── Train ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.85, random_state=SEED)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {acc:.2%}")
print(classification_report(y_test, y_pred, target_names=LABEL_ORDER))

cv_sc = cross_val_score(clf, X, y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring='accuracy')
print(f"5-Fold CV     : {cv_sc.mean():.2%} ± {cv_sc.std():.2%}")
print(f"Model features: {list(clf.feature_names_in_)}")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(clf, os.path.join(DIR, 'rf_model.pkl'))
joblib.dump(le,  os.path.join(DIR, 'label_encoder.pkl'))
with open(os.path.join(DIR, 'score_thresholds.json'), 'w') as f:
    json.dump({'Q_ATRISK': float(Q_ATRISK),
               'Q_AVERAGE': float(Q_AVERAGE),
               'Q_GOOD':    float(Q_GOOD)}, f, indent=2)

print("\nSaved: rf_model.pkl | label_encoder.pkl | score_thresholds.json")
print("Done — you can now run: python run.py")
