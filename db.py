"""
db.py — centralized PostgreSQL connection handling for Gradenix.

Replaces the old sqlite3 connection logic with psycopg2, reading all
credentials from environment variables (see .env.example).

Why a custom Row class?
------------------------
The original code (and the Jinja templates: teacher.html, admin.html,
student.html) relies on sqlite3.Row's dual behavior — a fetched row can be
read BOTH by column name (row["username"]) AND by position (row[0], row[1],
h[2][:16], etc.) and can also be unpacked with `*row` (User(*row)).

psycopg2's plain cursor returns plain tuples (no name access) and
RealDictCursor returns dicts (no positional access, and `*row` would unpack
the dict's KEYS instead of values). Neither on its own is a drop-in
replacement. So `Row` below wraps a tuple + the cursor's column names to
support both access styles without touching a single template or route.

get_db() is a context manager that:
  - opens a psycopg2 connection
  - commits on success
  - rolls back on any exception
  - always closes the connection
"""
import os
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

import psycopg2

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "gradenix")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "your_password")

# Render (and other PaaS providers) often expose a single DATABASE_URL
# instead of individual DB_* vars. Support both, preferring DATABASE_URL
# when present, without changing any application logic.
DATABASE_URL = os.environ.get("DATABASE_URL")


def _connect():
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


class Row(tuple):
    """A tuple that also supports column-name access and .get(), mirroring
    sqlite3.Row so existing code/templates (row[0], row["username"], *row)
    all keep working unmodified.
    """

    def __new__(cls, values, columns):
        obj = super().__new__(cls, values)
        obj._columns = columns
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                idx = self._columns.index(key)
            except ValueError:
                raise KeyError(key)
            return tuple.__getitem__(self, idx)
        return tuple.__getitem__(self, key)

    def keys(self):
        return list(self._columns)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _CursorWrapper:
    """Thin wrapper so `conn.execute(...)` keeps working like it did with
    sqlite3, instead of requiring conn.cursor().execute(...) everywhere.
    Also translates '?' placeholders (SQLite style) to '%s' (psycopg2 style)
    so the rest of the app's SQL strings did not need to be rewritten.
    """

    def __init__(self, conn):
        self._conn = conn
        self._cur = conn.cursor()

    def execute(self, query, params=None):
        query = query.replace("?", "%s")
        self._cur.execute(query, params or ())
        return self

    def _columns(self):
        return [d[0] for d in self._cur.description] if self._cur.description else []

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        return Row(row, self._columns())

    def fetchall(self):
        cols = self._columns()
        return [Row(r, cols) for r in self._cur.fetchall()]

    def __getattr__(self, item):
        return getattr(self._cur, item)


@contextmanager
def get_db():
    conn = _connect()
    wrapper = _CursorWrapper(conn)
    try:
        yield wrapper
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
