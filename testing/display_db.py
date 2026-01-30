import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "db" / "rooflytics.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

rows = cur.execute(
    "SELECT job_id, tile_name, num_roofs, hot_roofs, cool_roofs, energy_kwh, cost_nzd, co2_kg, created_at FROM analysis_results"
).fetchall()

for r in rows:
    print(r)

conn.close()