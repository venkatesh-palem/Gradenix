"""
run.py — start Gradenix
  python run.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load .env (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, ...) for local dev.
# On Render/production these are set as real environment variables, and
# load_dotenv() is a harmless no-op if no .env file exists.
from dotenv import load_dotenv
load_dotenv()

# DB init is now handled inside app.py on every startup — nothing to do here
from app import app

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=True)
