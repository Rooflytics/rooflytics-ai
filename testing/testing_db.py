import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
DB_PATH = ROOT / "db" / "rooflytics.db"

conn = sqlite3.connect(DB_PATH)
print(conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone())
