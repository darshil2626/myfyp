import time
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
from openaq import OpenAQ
from openaq.shared.exceptions import RateLimitError

# ---- simple config ----
API_KEY = "90b7cee0d53a58a19b6dda7b76183a1f35a2fbf2452cb72988005545b1b17c66"
LONDON_COORDS = (51.5074, -0.1278)
RADIUS_M = 1_000          # central London only
DAYS_BACK = 7
MAX_LOCATIONS = 8         # to avoid rate limits
MAX_SENSORS = 30
EXPECTED_MEASUREMENTS_PER_DAY = 24  # for completeness


def get_with_retry(func, *args, **kwargs):
    """Tiny helper to survive rate limits."""
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            print("Rate limit hit, sleeping 40s...")
            time.sleep(40)

client = OpenAQ(api_key=API_KEY)

# 1) Locations near London
locations = client.locations.list(
    coordinates=LONDON_COORDS,
    radius=RADIUS_M,
    limit=1000,
).dict()["results"]

locations = locations[:MAX_LOCATIONS]
locations_df = pd.json_normalize(locations)
print("Sample locations:\n", locations_df[["id", "name"]].head(), "\n")

# 2) Sensors for those locations
sensors = []
for loc in locations:
    loc_id = loc["id"]
    loc_name = loc["name"]
    print(f"Fetching sensors for {loc_name}")

    s_resp = get_with_retry(client.locations.sensors, locations_id=loc_id)
    for s in s_resp.dict()["results"]:
        param = s.get("parameter") or {}
        sensors.append({
            "location_id": loc_id,
            "location_name": loc_name,
            "sensor_id": s["id"],
            "parameter_name": param.get("name"),
            "parameter_units": param.get("units"),
        })

sensors_df = pd.DataFrame(sensors)
sensors_df = sensors_df.head(MAX_SENSORS)
print("\nSensors sample:\n", sensors_df.head(), "\n")

# 3) Measurements for last N days
end = datetime.utcnow()
start = end - timedelta(days=DAYS_BACK)
print(f"Fetching measurements from {start} to {end} (UTC)\n")

rows = []
for _, s in sensors_df.iterrows():
    print(f"  Sensor {s.sensor_id} ({s.location_name}, {s.parameter_name})")
    m = get_with_retry(
        client.measurements.list,
        sensors_id=int(s.sensor_id),
        datetime_from=start.isoformat(timespec="seconds") + "Z",
        datetime_to=end.isoformat(timespec="seconds") + "Z",
        limit=1000,
    ).dict()["results"]

    for r in m:
        period = r.get("period", {})
        dt_to = period.get("datetime_to", {})
        dt_utc = dt_to.get("utc")  # end of interval
        rows.append({
            "sensor_id": s.sensor_id,
            "location_id": s.location_id,
            "location_name": s.location_name,
            "parameter_name": s.parameter_name,
            "parameter_units": s.parameter_units,
            "value": r.get("value"),
            "datetime": dt_utc,
        })

client.close()

measurements_df = pd.DataFrame(rows)
measurements_df["datetime"] = pd.to_datetime(
    measurements_df["datetime"], errors="coerce"
)
print("\nMeasurements sample:\n", measurements_df.head(), "\n")

if measurements_df["datetime"].notna().sum() == 0:
    print("No valid datetimes returned.")

# 4) Plot one time series (first location + parameter with data)
sub = measurements_df.dropna(subset=["datetime"]).copy()
first_loc = sub["location_name"].iloc[0]
first_param = sub["parameter_name"].iloc[0]

plot_df = (
    sub[(sub["location_name"] == first_loc) &
        (sub["parameter_name"] == first_param)]
    .sort_values("datetime")
)

print(f"Plotting time series for {first_param} at {first_loc}\n")

plt.figure(figsize=(10, 4))
plt.plot(plot_df["datetime"], plot_df["value"])
plt.title(f"{first_param.upper()} at {first_loc}")
plt.xlabel("Time (UTC)")
plt.ylabel(f"{first_param.upper()} [{plot_df['parameter_units'].iloc[0]}]")
plt.tight_layout()
plt.show()

# 5) Daily completeness
measurements_df["date"] = measurements_df["datetime"].dt.date

daily_counts = (
    measurements_df
    .dropna(subset=["datetime"])
    .groupby(["location_name", "sensor_id", "parameter_name", "date"])
    .size()
    .reset_index(name="n_measurements")
)
daily_counts["completeness"] = (
    daily_counts["n_measurements"] / EXPECTED_MEASUREMENTS_PER_DAY
)

print("Daily counts sample:\n", daily_counts.head(), "\n")