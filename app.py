"""
app.py — Gradenix Flask application
"""
import hashlib, json, os, sys
from datetime import timedelta

import psycopg2

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# ── Resolve all paths relative to THIS file (never CWD-dependent) ─────────────
_BASE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _BASE)

TEMPLATE_FOLDER = _BASE
STATIC_FOLDER   = _BASE          # style.css lives flat next to app.py

# Load .env for local dev (no-op if missing / already loaded via run.py).
# Needed here too since gunicorn (Procfile: "gunicorn app:app") imports this
# module directly, bypassing run.py.
from dotenv import load_dotenv
load_dotenv()

from db import get_db

# ── Auto-init DB on every startup (idempotent — safe to call always) ──────────
from init_db import init as _init_db
_init_db()

# ── Auto-retrain guard ────────────────────────────────────────────────────────
_STALE = {"FamilyIncome", "CreditsCompleted"}
_REQUIRED = {
    "Attendance","StudyHours","PreviousSGPA","SkillDevelopmentHours",
    "SocialMediaHours","EnglishProficiency","Scholarship","CoCurricular",
    "HealthIssues","CurrentSemester","CurrentCGPA",
}

def _model_needs_retrain():
    pkl = os.path.join(_BASE, "rf_model.pkl")
    if not os.path.exists(pkl):
        return True
    try:
        import joblib
        m = joblib.load(pkl)
        trained = set(m.feature_names_in_)
        return bool(trained & _STALE) or not (_REQUIRED <= trained)
    except Exception:
        return True

if _model_needs_retrain():
    print("[Gradenix] Model missing or stale — retraining…")
    import retrain_model  # noqa: executes on import
    print("[Gradenix] Retrain complete.")

from model import predict, get_tips

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER,
    static_url_path="/static",
)

# Strong secret key — reads from env var in production, falls back to a fixed dev key
app.secret_key = os.environ.get("GRADENIX_SECRET", "gx-dev-secret-change-in-prod-2025")

# Keep sessions alive across browser closes (7-day rolling session)
app.config["REMEMBER_COOKIE_DURATION"] = timedelta(days=7)
app.config["REMEMBER_COOKIE_SECURE"]   = False   # set True behind HTTPS
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["SESSION_PERMANENT"]        = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

login_manager = LoginManager(app)
login_manager.login_view     = "login"
login_manager.login_message  = "Please sign in to continue."
login_manager.login_message_category = "error"


# ── Auth ──────────────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, role, full_name=""):
        self.id        = id
        self.username  = username
        self.role      = role
        self.full_name = full_name

@login_manager.user_loader
def load_user(uid):
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT id,username,role,full_name FROM users WHERE id=?", (uid,)
            ).fetchone()
        return User(*row) if row else None
    except Exception:
        return None

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


# ── Routes: root ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}_dashboard"))
    return redirect(url_for("login"))


# ── Routes: login ─────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("login.html")

        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT id,username,role,full_name FROM users WHERE username=? AND password=?",
                    (username, hash_pw(password))
                ).fetchone()
        except psycopg2.OperationalError as e:
            # e.g. connection dropped/unavailable — don't crash, tell the user to retry
            print(f"[Gradenix] DB error during login: {e}")
            flash("The server is busy, please try again in a moment.", "error")
            return render_template("login.html")

        if row:
            user = User(*row)
            login_user(user, remember=True)   # remember=True → 7-day cookie
            return redirect(url_for(f"{row['role']}_dashboard"))

        flash("Incorrect username or password.", "error")

    return render_template("login.html")


# ── Routes: logout ────────────────────────────────────────────────────────────
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── Routes: signup ────────────────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}_dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username  = request.form.get("username",  "").strip()
        password  = request.form.get("password",  "")
        confirm   = request.form.get("confirm_password", "")

        # Validate
        error = None
        if not full_name or not username or not password:
            error = "All fields are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."

        if error:
            flash(error, "error")
            return render_template("signup.html")

        # Persist — use try/except only for IntegrityError (duplicate username)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username,password,role,full_name) VALUES (?,?,?,?)",
                    (username, hash_pw(password), "student", full_name)
                )
                # Fetch the newly created user in the same connection
                row = conn.execute(
                    "SELECT id,username,role,full_name FROM users WHERE username=?",
                    (username,)
                ).fetchone()

            # Log in immediately and set persistent cookie
            login_user(User(row["id"], row["username"], row["role"], row["full_name"]), remember=True)
            flash(f"Welcome, {full_name}! Your account has been created.", "success")
            return redirect(url_for("student_dashboard"))

        except psycopg2.errors.UniqueViolation:
            flash("That username is already taken. Please choose another.", "error")
        except psycopg2.OperationalError as e:
            print(f"[Gradenix] DB error during signup: {e}")
            flash("The server is busy, please try again in a moment.", "error")

    return render_template("signup.html")


# ── Routes: student ───────────────────────────────────────────────────────────
@app.route("/student", methods=["GET", "POST"])
@login_required
def student_dashboard():
    if current_user.role != "student":
        return redirect(url_for("login"))

    result = confidence = None
    tips      = []
    form_data = {}

    if request.method == "POST":
        try:
            form_data = dict(request.form)

            features = {
                "CurrentCGPA":           float(request.form["CurrentCGPA"]),
                "Attendance":            float(request.form["Attendance"]),
                "StudyHours":            float(request.form["StudyHours"]),
                "SocialMediaHours":      float(request.form["SocialMediaHours"]),
                "PreviousSGPA":          float(request.form["PreviousSGPA"]),
                "SkillDevelopmentHours": float(request.form["SkillDevelopmentHours"]),
                "CoCurricular":          int(request.form.get("CoCurricular", 0)),
                "HealthIssues":          int(request.form.get("HealthIssues", 0)),
                # Non-form features — sensible defaults
                "EnglishProficiency":    1,
                "Scholarship":           0,
                "CurrentSemester":       4,
            }

            result, confidence = predict(features)
            tips = get_tips(features)

            with get_db() as conn:
                conn.execute(
                    "INSERT INTO predictions (user_id,result,confidence,tips) VALUES (?,?,?,?)",
                    (current_user.id, result, confidence, json.dumps(tips))
                )

        except KeyError as e:
            flash(f"Missing field: {e}. Please fill in all inputs.", "error")
        except ValueError as e:
            flash(f"Invalid value: {e}. Please enter numbers only.", "error")

    # Always fetch history in a fresh connection block
    with get_db() as conn:
        history = conn.execute(
            """SELECT result, confidence, TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
               FROM predictions
               WHERE user_id=? ORDER BY created_at DESC LIMIT 5""",
            (current_user.id,)
        ).fetchall()

    return render_template(
        "student.html",
        result=result, confidence=confidence, tips=tips,
        history=history, form_data=form_data
    )


# ── Routes: teacher ───────────────────────────────────────────────────────────
@app.route("/teacher")
@login_required
def teacher_dashboard():
    if current_user.role != "teacher":
        return redirect(url_for("login"))

    with get_db() as conn:
        students = conn.execute("""
            SELECT u.username, u.full_name, p.result, p.confidence,
                   TO_CHAR(p.created_at, 'YYYY-MM-DD HH24:MI:SS') AS created_at
            FROM predictions p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.created_at DESC
        """).fetchall()
        stats = conn.execute(
            "SELECT result, COUNT(*) as cnt FROM predictions GROUP BY result"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    return render_template("teacher.html", students=students, stats=stats, total=total)


# ── Routes: admin ─────────────────────────────────────────────────────────────
@app.route("/admin")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        return redirect(url_for("login"))

    with get_db() as conn:
        users       = conn.execute("SELECT id,username,role,full_name FROM users ORDER BY role,username").fetchall()
        total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        stats       = conn.execute("SELECT result,COUNT(*) FROM predictions GROUP BY result").fetchall()

    return render_template("admin.html", users=users,
                           total_preds=total_preds, total_users=total_users, stats=stats)


@app.route("/admin/add_user", methods=["POST"])
@login_required
def add_user():
    if current_user.role != "admin":
        return redirect(url_for("login"))

    username  = request.form.get("username",  "").strip()
    password  = request.form.get("password",  "")
    role      = request.form.get("role",      "student")
    full_name = request.form.get("full_name", "").strip()

    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for("admin_dashboard"))

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username,password,role,full_name) VALUES (?,?,?,?)",
                (username, hash_pw(password), role, full_name)
            )
        flash(f"User '{username}' ({role}) added successfully.", "success")
    except psycopg2.errors.UniqueViolation:
        flash(f"Username '{username}' already exists.", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete_user/<int:uid>", methods=["POST"])
@login_required
def delete_user(uid):
    if current_user.role != "admin":
        return redirect(url_for("login"))

    if uid == int(current_user.id):
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin_dashboard"))

    with get_db() as conn:
        # foreign_keys=ON + ON DELETE CASCADE handles predictions automatically
        conn.execute("DELETE FROM users WHERE id=?", (uid,))

    flash("User and their predictions deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/reset_db", methods=["POST"])
@login_required
def reset_db():
    if current_user.role != "admin":
        return redirect(url_for("login"))

    with get_db() as conn:
        conn.execute("DELETE FROM predictions")

    flash("All prediction records cleared. User accounts are intact.", "success")
    return redirect(url_for("admin_dashboard"))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=True)
