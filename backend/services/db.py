import sqlite3
from pathlib import Path

DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)

DB_PATH = DB_DIR / "rooflytics.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    with open(DB_DIR / "schema.sql", "r") as f:
        cur.executescript(f.read())

    conn.commit()
    conn.close()


def insert_analysis_result(
    job_id: str,
    tile_name: str,
    num_roofs: int,
    hot_roofs: int,
    cool_roofs: int,
    energy_kwh: float,
    cost_nzd: float,
    co2_kg: float,
    usage_factor: float,
    max_kwh_per_roof: float,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO analysis_results (
            job_id, tile_name, num_roofs, hot_roofs, cool_roofs,
            energy_kwh, cost_nzd, co2_kg, usage_factor, max_kwh_per_roof
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            tile_name,
            num_roofs,
            hot_roofs,
            cool_roofs,
            energy_kwh,
            cost_nzd,
            co2_kg,
            usage_factor,
            max_kwh_per_roof,
        ),
    )

    conn.commit()
    conn.close()
