import pandas as pd, numpy as np, glob, os, time

SRC_DIR = 'data/PV_data'
DST_DIR = 'data/PV_data_trafo'
os.makedirs(DST_DIR, exist_ok=True)

PLACEHOLDERS = {
    'WIND_DEGREES_100M_SIN': -0.211349,
    'WIND_DEGREES_100M_COS': -0.179375,
    'WIND_SPEED_100M':        0.199177,
}

FINAL_COLS = [
    'WIND_DEGREES_40M_SIN', 'WIND_DEGREES_40M_COS', 'PRESSURE_SURFACE',
    'DIRECT_HORIZONTAL_IRRADIANCE', 'WIND_SPEED_40M',
    'TEMPERATURE_2METER_ABOVE_GROUND', 'HUMIDITY_RELATIVE',
    'DIFFUSE_HORIZONTAL_IRRADIANCE', 'SOLAR_ELEVATION',
    'SOLAR_AZIMUTH_SIN', 'SOLAR_AZIMUTH_COS',
    'TIME_OF_DAY_SIN', 'TIME_OF_DAY_COS',
    'DAY_OF_YEAR_SIN', 'DAY_OF_YEAR_COS',
    'WIND_DEGREES_100M_SIN', 'WIND_DEGREES_100M_COS', 'WIND_SPEED_100M',
    'active_power_kW'
]

files   = glob.glob(f'{SRC_DIR}/*.parquet')
n       = len(files)
failed  = []
t0      = time.time()

print(f'Processing {n} files...')

for i, f in enumerate(files):
    fname = os.path.basename(f)
    out   = f'{DST_DIR}/{fname}'

    if os.path.exists(out):
        continue

    try:
        df = pd.read_parquet(f)
        for col, val in PLACEHOLDERS.items():
            df[col] = val
        df[FINAL_COLS].to_parquet(out)
    except Exception as e:
        failed.append((fname, str(e)))

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        eta     = (n - i - 1) / ((i + 1) / elapsed) / 60
        print(f'  {i+1}/{n} | {elapsed:.0f}s | ETA {eta:.1f} min')

print(f'\nDone in {(time.time()-t0)/60:.1f} min | failed: {len(failed)}')
if failed:
    for fname, err in failed:
        print(f'  {fname}: {err}')

