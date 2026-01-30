CREATE TABLE IF NOT EXISTS analysis_results (
    job_id            VARCHAR(36) PRIMARY KEY,
    tile_name         VARCHAR(255),
    num_roofs         INT,
    hot_roofs         INT,
    cool_roofs        INT,
    energy_kwh        DOUBLE,
    cost_nzd          DOUBLE,
    co2_kg            DOUBLE,
    usage_factor      DOUBLE,
    max_kwh_per_roof  DOUBLE,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
