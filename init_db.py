"""
init_db.py — creates the PostgreSQL schema and seed accounts.
Safe to run multiple times: uses CREATE TABLE IF NOT EXISTS and
ON CONFLICT DO NOTHING for seed rows.
Called automatically by run.py and app.py on first startup.
"""
import hashlib

from db import get_db


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def init():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        SERIAL PRIMARY KEY,
            username  TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'student',
            full_name TEXT NOT NULL DEFAULT ''
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id         SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            result     TEXT NOT NULL,
            confidence REAL NOT NULL,
            tips       TEXT NOT NULL DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # ON CONFLICT DO NOTHING — never overwrites existing accounts
        seeds = [
            ("admin",    hash_pw("admin123"),   "admin",   "Administrator"),
            ("teacher1", hash_pw("teacher123"), "teacher", "Prof. Ramesh"),
        ]
        for username, pw, role, name in seeds:
            conn.execute(
                "INSERT INTO users (username,password,role,full_name) "
                "VALUES (?,?,?,?) ON CONFLICT (username) DO NOTHING",
                (username, pw, role, name)
            )

    print("[Gradenix] PostgreSQL DB ready (tables created if missing).")
    print("[Gradenix] Default accounts: admin/admin123  |  teacher1/teacher123")


if __name__ == "__main__":
    init()
