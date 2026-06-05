import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_URL = "https://api.erg.ic.ac.uk/AirQuality"


def get_london_sites():
    """
    Get all monitoring sites in the 'London' group as a DataFrame.
    """
    url = f"{BASE_URL}/Information/MonitoringSites/GroupName=London/Json"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    sites_list = data.get("Sites", {}).get("Site", [])
    sites_df = pd.json_normalize(sites_list)

    # Standardise SiteCode column
    if "@SiteCode" in sites_df.columns and "SiteCode" not in sites_df.columns:
        sites_df["SiteCode"] = sites_df["@SiteCode"]

    return sites_df


def _parse_columns_metadata(aq):
    """
    Robustly parse AirQualityData['Columns']['Column'] into a dict:
        Data1 -> 'Nitric Oxide (ug/m3)', etc.

    Handles cases where Column is a dict, list of dicts, or weird strings.
    """
    cols_meta = aq.get("Columns", {}).get("Column", [])

    # Normalise to list
    if isinstance(cols_meta, dict):
        cols_meta = [cols_meta]
    elif isinstance(cols_meta, str):
        # Nothing useful here
        return {}

    id_to_name = {}
    for c in cols_meta:
        # Skip anything that isn't a dict
        if not isinstance(c, dict):
            continue
        col_id = c.get("@ColumnId")
        col_name = c.get("@ColumnName")
        if col_id and col_name:
            id_to_name[col_id] = col_name

    return id_to_name


def fetch_site_hourly_wide(site_code, start_date, end_date):
    """
    Fetch hourly wide data for one site using /Data/Wide/Site/...

    Returns a DataFrame with:
      - datetime
      - SiteCode
      - one column per pollutant (nicely named, e.g. 'Nitric Oxide (ug/m3)')
    """
    url = (
        f"{BASE_URL}/Data/Wide/Site/"
        f"SiteCode={site_code}/"
        f"StartDate={start_date}/"
        f"EndDate={end_date}/Json"
    )

    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    aq = data.get("AirQualityData")
    if aq is None:
        print(f"No 'AirQualityData' in response for {site_code}")
        print("Raw response:", data)
        return pd.DataFrame()

    # --- robust metadata parsing ---
    id_to_name = _parse_columns_metadata(aq)

    raw = aq.get("RawAQData", {})
    rows = raw.get("Data", [])

    if not rows:
        print(f"No rows for site {site_code} between {start_date} and {end_date}")
        return pd.DataFrame()

    df = pd.json_normalize(rows)

    # Parse datetime
    if "@MeasurementDateGMT" in df.columns:
        df["datetime"] = pd.to_datetime(df["@MeasurementDateGMT"], errors="coerce")

    df["SiteCode"] = site_code

    # Replace empty strings with NaN (and clean dtypes)
    df = df.replace("", np.nan)
    df = df.infer_objects(copy=False)  # avoids the FutureWarning

    # Rename @Data1..@DataN to pretty pollutant names
    rename_map = {}
    for col in df.columns:
        if col.startswith("@Data"):
            col_id = col.lstrip("@")  # 'Data1'
            pretty_name = id_to_name.get(col_id, col_id)
            # strip leading "- " and site name; keep just pollutant + units
            pretty_name = pretty_name.lstrip("- ").split(": ", 1)[-1]
            rename_map[col] = pretty_name

    df = df.rename(columns=rename_map)

    return df


def fetch_all_sites_hourly_wide(start_date, end_date, max_sites=None):
    """
    Fetch hourly wide data for multiple London sites and concatenate into one DataFrame.

    max_sites: limit number of sites (for testing). Set to None for all sites.
    """
    sites_df = get_london_sites()
    if max_sites is not None:
        sites_df = sites_df.head(max_sites)

    all_dfs = []

    for _, row in sites_df.iterrows():
        site_code = row.get("SiteCode") or row.get("@SiteCode")
        if not site_code:
            continue

        print(f"Fetching site {site_code} ...")
        try:
            df_site = fetch_site_hourly_wide(site_code, start_date, end_date)
        except Exception as e:
            # Don't let one weird site kill the whole run
            print(f"  Error fetching {site_code}: {e}")
            continue

        if not df_site.empty:
            all_dfs.append(df_site)

    if not all_dfs:
        print("No data returned for any site.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


def completeness_by_site(df, pollutant_col):
    """
    Fraction of non-missing values per site for a given pollutant column.
    """
    if pollutant_col not in df.columns:
        raise ValueError(f"{pollutant_col} not in DataFrame columns: {list(df.columns)}")

    return (
        df.groupby("SiteCode")[pollutant_col]
          .apply(lambda x: x.notna().mean())
          .sort_values(ascending=False)
    )

def get_pollutant_columns(df):
    """
    Heuristic to pick pollutant columns from the DataFrame.
    Excludes datetime / key columns, and keeps anything that looks like
    a concentration with units (ug/m3, mg/m3) or is numeric.
    """
    exclude = {"datetime", "SiteCode", "@MeasurementDateGMT"}
    cols = []

    for c in df.columns:
        if c in exclude:
            continue
        # Likely pollutant if it has units in the name
        if "(ug/m3)" in c or "(mg/m3)" in c:
            cols.append(c)
            continue
        # Or if it's numeric and not one of the excluded keys
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    return cols


def completeness_matrix(df, pollutant_cols=None):
    """
    Compute completeness (fraction of non-missing values) for
    every (site, pollutant) pair.

    Returns a DataFrame with:
      index  = SiteCode
      columns = pollutant names
      values = completeness in [0, 1]
    """
    if "SiteCode" not in df.columns:
        raise ValueError("DataFrame must contain a 'SiteCode' column")

    if pollutant_cols is None:
        pollutant_cols = get_pollutant_columns(df)

    if not pollutant_cols:
        raise ValueError("No pollutant columns found to compute completeness.")

    # For each pollutant, compute completeness by site
    comp_dfs = []
    for pollutant in pollutant_cols:
        comp = (
            df.groupby("SiteCode")[pollutant]
              .apply(lambda x: x.notna().mean())
        )
        comp_dfs.append(comp.rename(pollutant))

    # Combine into a single matrix
    comp_matrix = pd.concat(comp_dfs, axis=1)

    return comp_matrix


def overall_site_completeness(comp_matrix):
    """
    Given the matrix from completeness_matrix, compute an overall
    completeness per site (mean across pollutants).
    """
    return comp_matrix.mean(axis=1).sort_values(ascending=False)


if __name__ == "__main__":
    # ----- choose a time window (last 70 days, starting 170 days ago as example) -----
    days_back = 700
    start_back = 1000
    start = (datetime.utcnow() - timedelta(days=(days_back + start_back))).strftime("%Y-%m-%d")
    end = (datetime.utcnow() - timedelta(days=start_back)).strftime("%Y-%m-%d")

    print(f"Using date range: {start} to {end}")

    # ----- list sites -----
    sites_df = get_london_sites()
    print("Number of London sites:", len(sites_df))
    print(sites_df[["SiteCode"]].head())

    # ----- example site: first one -----
    example_site = sites_df.iloc[0]["SiteCode"]
    print("\nExample site:", example_site)

    df_example = fetch_site_hourly_wide(example_site, start, end)
    print("\nExample site data (head):")
    print(df_example.head())

    print("\nMissing fraction per column for example site:")
    print(df_example.isna().mean())

    # ----- fetch multiple sites for a quick data-quality overview -----
    print("\nFetching multiple sites (max_sites=10) ...")
    full_df = fetch_all_sites_hourly_wide(start, end, max_sites=10)

    print("\nCombined DataFrame shape:", full_df.shape)
    print("Columns:")
    print(full_df.columns)
    
    if not full_df.empty:
        # 1) Build completeness matrix
        pollutant_cols = get_pollutant_columns(full_df)
        print("\nDetected pollutant columns:", pollutant_cols)

        comp_matrix = completeness_matrix(full_df, pollutant_cols=pollutant_cols)
        print("\nCompleteness matrix (head):")
        print(comp_matrix.head())

        # 2) Overall completeness per site
        overall_comp = overall_site_completeness(comp_matrix)
        print("\nOverall completeness per site (top 10):")
        print(overall_comp.head(10))

        print("\nOverall completeness per site (bottom 10):")
        print(overall_comp.tail(10))

        # 3) Optional: save to CSV for inspection / LaTeX
        comp_matrix.to_csv("london_air_completeness_matrix.csv")
        overall_comp.to_csv("london_air_overall_completeness.csv", header=["overall_completeness"])
    else:
        print("full_df is empty; no completeness summary computed.")


    # Pick a pollutant column (any with units in the name, as per metadata)
    candidate_pollutants = [c for c in full_df.columns if "(ug/m3)" in c]
    if candidate_pollutants:
        pollutant = candidate_pollutants[0]
        print(f"\nUsing pollutant for completeness check: {pollutant}")
        comp = completeness_by_site(full_df, pollutant)
        print("\nFraction of available data per site (top 10):")
        print(comp.head(10))
        print("\nBottom 10 sites by completeness:")
        print(comp.tail(10))
    else:
        print("\nNo pollutant columns found with '(ug/m3)' to compute completeness.")
