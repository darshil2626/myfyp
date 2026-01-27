"""
copernicus_s5p_no2_download.py

Search & download Sentinel-5P L2 NO2 products from the
Copernicus Data Space Ecosystem using the OData API.

- Documentation:
  * OData basics & examples: https://documentation.dataspace.copernicus.eu/notebook-samples/geo/odata_basics.html
  * Migration guide (shows token + zipper download): https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/migration_from_scihub_guide.html
"""

import os
import json
import getpass
from pathlib import Path
from typing import Tuple, List

import requests
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG – EDIT THESE FOR YOUR USE CASE
# ---------------------------------------------------------------------

# Area of interest: rough bounding box around Greater London (lon, lat)
# WKT POLYGON is lon lat, lon lat, ...
AOI_WKT = (
    "POLYGON(("
    "-0.6 51.2,"
    "0.4 51.2,"
    "0.4 51.8,"
    "-0.6 51.8,"
    "-0.6 51.2"
    "))"
)

# Time range (UTC, date only; times will be 00:00:00Z and 23:59:59Z)
START_DATE = "2024-01-01"
END_DATE = "2024-01-03"

# Sentinel-5P L2 NO2 offline product type
COLLECTION_NAME = "SENTINEL-5P"
PRODUCT_TYPE = "L2__NO2___"  # see S5P docs for other types

# Max number of products to fetch (for testing, keep it small)
MAX_PRODUCTS = 5

# Output directory for downloaded ZIPs
OUTPUT_DIR = "data_s5p_no2"


# ---------------------------------------------------------------------
# CONSTANT ENDPOINTS
# ---------------------------------------------------------------------

CATALOGUE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ZIPPER_ODATA_URL = "https://zipper.dataspace.copernicus.eu/odata/v1"
AUTH_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


# ---------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------

def get_access_token(username: str, password: str) -> str:
    """
    Get an OAuth2 access token for the Copernicus Data Space Ecosystem.

    This follows the official example:
    - client_id: cdse-public
    - grant_type: password
    https://documentation.dataspace.copernicus.eu/notebook-samples/geo/odata_basics.html
    """
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(AUTH_URL, data=data)
    try:
        r.raise_for_status()
    except Exception:
        raise RuntimeError(
            f"Token request failed ({r.status_code}): {r.text}"
        )
    resp = r.json()
    if "access_token" not in resp:
        raise RuntimeError(f"No access_token in response: {resp}")
    return resp["access_token"]


def get_credentials() -> Tuple[str, str]:
    """
    Get CDSE username/password from env vars or prompt the user.
    """
    username = "darshilgajjar2002@gmail.com"
    password = "Swamishri22!"

    if not username:
        username = input("CDSE username: ")
    if not password:
        password = getpass.getpass("CDSE password: ")

    return username, password


# ---------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------

def build_search_url(
    collection_name: str,
    product_type: str,
    aoi_wkt: str,
    start_date: str,
    end_date: str,
    top: int = None,
) -> str:
    """
    Build an OData /Products query URL for Sentinel-5P.

    Pattern based on official OData examples:
    https://documentation.dataspace.copernicus.eu/notebook-samples/geo/odata_basics.html
    and migration guide.
    """
    start_iso = f"{start_date}T00:00:00.000Z"
    end_iso = f"{end_date}T23:59:59.999Z"

    filter_expr = (
        f"Collection/Name eq '{collection_name}' "
        f"and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and "
        f"att/OData.CSC.StringAttribute/Value eq '{product_type}'"
        f") "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start gt {start_iso} "
        f"and ContentDate/Start lt {end_iso}"
    )

    base = f"{CATALOGUE_ODATA_URL}/Products?$filter={filter_expr}"
    if top is not None:
        base += f"&$top={top}"
    return base


def search_s5p_products(
    collection_name: str,
    product_type: str,
    aoi_wkt: str,
    start_date: str,
    end_date: str,
    top: int = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Query the catalogue for Sentinel-5P products.

    Returns:
        - DataFrame with catalogue entries
        - Raw JSON list (for IDs etc.)
    """
    url = build_search_url(
        collection_name, product_type, aoi_wkt, start_date, end_date, top
    )

    print("Search URL:\n", url.replace(" ", "%20"), "\n")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    values = data.get("value", [])
    if not values:
        print("No products found.")
        return pd.DataFrame(), []

    df = pd.DataFrame.from_dict(values)
    return df, values


# ---------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------

def download_product_zip(
    product_id: str,
    access_token: str,
    out_dir: str,
    suggested_name: str = None,
) -> Path:
    """
    Download a full product ZIP via the zipper OData endpoint.

    Example pattern (from migration guide):
    https://zipper.dataspace.copernicus.eu/odata/v1/Products(<ID>)/$value
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Build download URL
    url = f"{ZIPPER_ODATA_URL}/Products({product_id})/$value"

    headers = {"Authorization": f"Bearer {access_token}"}
    session = requests.Session()
    session.headers.update(headers)

    print(f"Downloading product {product_id} ...")
    r = session.get(url, stream=True)
    r.raise_for_status()

    # Decide filename
    if suggested_name:
        filename = suggested_name
    else:
        filename = f"{product_id}.zip"

    out_path = Path(out_dir) / filename

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=== Sentinel-5P L2 NO2 download via Copernicus Data Space ===")

    # 1) Search catalogue
    df, products = search_s5p_products(
        collection_name=COLLECTION_NAME,
        product_type=PRODUCT_TYPE,
        aoi_wkt=AOI_WKT,
        start_date=START_DATE,
        end_date=END_DATE,
        top=MAX_PRODUCTS,
    )

    if df.empty:
        return

    # Show quick summary
    cols_to_show = ["Id", "Name", "ContentDate"]
    print("\nFound products:")
    print(df[cols_to_show].head())

    # 2) Auth & token (needed only for download, not for search)
    username, password = get_credentials()
    token = get_access_token(username, password)
    print("\nGot access token.")

    # 3) Download each product (as ZIP)
    for i, row in df.iterrows():
        product_id = row["Id"]
        product_name = row["Name"]
        # zip name derived from product name
        zip_name = f"{product_name}.zip"
        download_product_zip(
            product_id=product_id,
            access_token=token,
            out_dir=OUTPUT_DIR,
            suggested_name=zip_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
