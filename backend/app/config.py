from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
EXPERIMENTS_DIR = Path(os.getenv("EXPERIMENTS_DIR", BASE_DIR / "experiments"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ML defaults
TILE_SIZE = int(os.getenv("TILE_SIZE", 512))
DEFAULT_BACKBONE = os.getenv("DEFAULT_BACKBONE", "efficientnet")

# Energy model constants
SOLAR_IRRADIANCE = float(os.getenv("SOLAR_IRRADIANCE", 0.75))
SUNLIGHT_HOURS = float(os.getenv("SUNLIGHT_HOURS", 1700))
COOLING_EFFICIENCY = float(os.getenv("COOLING_EFFICIENCY", 0.65))
ELECTRICITY_PRICE = float(os.getenv("ELECTRICITY_PRICE", 0.3))
EMISSION_FACTOR = float(os.getenv("EMISSION_FACTOR", 0.1))