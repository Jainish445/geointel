"""
GDELT Data Collector
====================
Fetches country-level geopolitical event data from GDELT 1.0 daily exports.
Falls back to mock data if GDELT is unavailable or yields 0 valid events.

Root-cause notes (bugs fixed):
1. ENCODING: GDELT files are Latin-1, not UTF-8. Use encoding='latin-1'.
2. COLUMN COUNT: GDELT 1.0 has 58 columns, not 44. The original column list
   skipped 14 Actor sub-fields (KnownGroupCode, EthnicCode, Religion1/2Code,
   Type1/2/3Code) between Actor1 and Actor2. This misaligned ALL column names
   after col 7, mapping 'Actor2CountryCode' → col 10 (Actor1Religion1Code),
   which is always blank → 0 bilateral events.
3. nrows: Must read the full file first, then filter for bilateral events.
   The first N rows are mostly sub-national/non-state actors with blank country codes.
"""

import pandas as pd
import requests
import io
import random
from datetime import datetime, timedelta
from typing import Optional


# ── GDELT 1.0 complete 58-column schema ─────────────────────────────────────
GDELT_COLS = [
    # 0-4
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    # 5-14  Actor1 (10 fields)
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    # 15-24  Actor2 (10 fields)
    "Actor2Code", "Actor2Name", "Actor2CountryCode",         # <-- col 17, not 10
    "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    # 25-34  Event
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone",
    # 35-41  Actor1 Geo
    "Actor1Geo_Type", "Actor1Geo_Fullname", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    # 42-48  Actor2 Geo
    "Actor2Geo_Type", "Actor2Geo_Fullname", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    # 49-55  Action Geo
    "ActionGeo_Type", "ActionGeo_Fullname", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    # 56-57
    "DATEADDED", "SOURCEURL",
]
assert len(GDELT_COLS) == 58, f"Expected 58 GDELT cols, got {len(GDELT_COLS)}"

# CAMEO QuadClass mapping
QUAD_CLASS_MAP = {
    1: "Verbal Cooperation",
    2: "Material Cooperation",
    3: "Verbal Conflict",
    4: "Material Conflict"
}

# CAMEO EventRootCode → label
EVENT_ROOT_MAP = {
    "01": "Make Public Statement",
    "02": "Appeal",
    "03": "Express Intent to Cooperate",
    "04": "Consult",
    "05": "Engage in Diplomatic Cooperation",
    "06": "Engage in Material Cooperation",
    "07": "Provide Aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit Force Posture",
    "16": "Reduce Relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use Unconventional Mass Violence",
}


def get_gdelt_url(date: datetime) -> str:
    return f"http://data.gdeltproject.org/events/{date.strftime('%Y%m%d')}.export.CSV.zip"


def fetch_gdelt_day(date: datetime, target_rows: int = 2000) -> Optional[pd.DataFrame]:
    """
    Download one day's GDELT file, extract all rows where BOTH actors have
    valid 3-letter ISO country codes, then return up to target_rows of them.

    Why no nrows limit: GDELT sorts events by ID, not actor type. Country-to-
    country events are scattered throughout the file; using nrows on the first
    10k rows almost always misses them entirely.
    """
    url = get_gdelt_url(date)
    try:
        print(f"  Fetching GDELT: {url}")
        resp = requests.get(url, timeout=90, stream=True)
        resp.raise_for_status()

        # Collect raw ZIP bytes without any text decoding
        chunks = []
        for chunk in resp.iter_content(chunk_size=131072):
            if chunk:
                chunks.append(chunk)
        raw_bytes = b"".join(chunks)

        # Read full file with correct 58-column schema + Latin-1 encoding
        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            header=None,
            names=GDELT_COLS,
            dtype=str,
            encoding="latin-1",
            on_bad_lines="skip",
            compression="zip",
        )

        # Filter to bilateral country-level events
        c1 = df["Actor1CountryCode"].fillna("").str.strip()
        c2 = df["Actor2CountryCode"].fillna("").str.strip()
        mask = (
            (c1.str.len() == 3) & (c2.str.len() == 3) &
            (c1 != "") & (c2 != "") & (c1 != c2)
        )
        df = df[mask].copy()

        total = len(df)
        if total == 0:
            print(f"  ✓ {date.strftime('%Y-%m-%d')}: 0 bilateral events in file")
            return None

        # Sample down if we have more than needed
        if total > target_rows:
            df = df.sample(n=target_rows, random_state=42)

        print(f"  ✓ {date.strftime('%Y-%m-%d')}: {total:,} bilateral events "
              f"(keeping {len(df):,})")
        return df

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        raise
    except Exception as e:
        print(f"  GDELT fetch failed ({date.strftime('%Y-%m-%d')}): {e}")
        return None


def collect_gdelt_range(
    start: datetime,
    end: datetime,
    target_rows_per_day: int = 2000,
) -> pd.DataFrame:
    """
    Collect GDELT data across a date range.
    Falls back to mock data if no valid events are collected.
    """
    frames = []
    current = start
    try:
        while current <= end:
            df = fetch_gdelt_day(current, target_rows_per_day)
            if df is not None:
                df["date"] = current.strftime("%Y-%m-%d")
                frames.append(df)
            current += timedelta(days=1)
    except KeyboardInterrupt:
        print("\nGDELT collection interrupted — using data collected so far.")

    if not frames:
        print("No GDELT bilateral events found. Falling back to mock data.")
        return generate_mock_data(start, end)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nGDELT collected: {len(combined):,} bilateral events "
          f"across {len(frames)} days.")
    return combined


def generate_mock_data(start: datetime, end: datetime, n_events: int = 5000) -> pd.DataFrame:
    """Realistic mock geopolitical event data for offline testing."""
    # All 163 GDELT-compatible ISO-3 country codes
    countries = [
        # Major powers
        "USA","CHN","RUS","DEU","GBR","FRA","IND","BRA","JPN","KOR",
        "ISR","IRN","SAU","TUR","PAK","NGA","ZAF","EGY","MEX","ARG",
        "IDN","AUS","CAN","ITA","UKR","POL","NLD","SWE","NOR","CHL",
        # Europe
        "ESP","PRT","BEL","AUT","CHE","DNK","FIN","GRC","CZE","HUN",
        "ROU","BGR","HRV","SVK","SVN","SRB","BLR","MDA","ALB","LTU",
        "LVA","EST","BIH","MKD","MNE","IRL","LUX",
        # Middle East & Central Asia
        "IRQ","SYR","JOR","LBN","YEM","OMN","ARE","QAT","KWT","BHR",
        "AFG","KAZ","UZB","TKM","TJK","KGZ","AZE","ARM","GEO",
        # Asia-Pacific
        "VNM","THA","MYS","PHL","SGP","BGD","LKA","NPL","MMR","KHM",
        "LAO","MNG","NZL","PNG","TWN","PRK",
        # Africa
        "ETH","TZA","KEN","GHA","CIV","AGO","CMR","MOZ","MDG","ZMB",
        "ZWE","SEN","MWI","MLI","BFA","NER","TCD","SDN","LBY","TUN",
        "DZA","MAR","COD","UGA","RWA","SOM","DJI","GAB","COG","TGO",
        "BEN","LBR","SLE","GIN","MRT","NAM","BWA",
        # Americas
        "COL","VEN","PER","ECU","BOL","PRY","URY","GTM","HND","SLV",
        "NIC","CRI","PAN","CUB","DOM","HTI","JAM","TTO","GUY","SUR",
        # Oceania
        "FJI",
    ]
    # Expanded tension pairs (negative tone bias)
    tensions = {
        ("USA","CHN"),("USA","RUS"),("USA","IRN"),("USA","PRK"),("USA","CUB"),
        ("CHN","IND"),("CHN","TWN"),("CHN","JPN"),("CHN","VNM"),("CHN","PHL"),
        ("RUS","UKR"),("RUS","POL"),("RUS","LTU"),("RUS","LVA"),("RUS","EST"),
        ("RUS","GEO"),("RUS","AZE"),("ISR","IRN"),("ISR","SYR"),("ISR","LBN"),
        ("IND","PAK"),("SAU","IRN"),("SAU","YEM"),("ETH","ERI"),("ETH","SOM"),
        ("SDN","SSD"),("SYR","TUR"),("ARM","AZE"),("GRC","TUR"),("SRB","XKX"),
        ("IRQ","SYR"),("AFG","PAK"),("IND","CHN"),
    }
    # Expanded alliance pairs (positive tone bias)
    alliances = {
        ("USA","GBR"),("USA","DEU"),("USA","JPN"),("USA","AUS"),("USA","CAN"),
        ("USA","KOR"),("USA","ISR"),("USA","FRA"),("USA","ITA"),("USA","ESP"),
        ("USA","NLD"),("USA","POL"),("USA","NOR"),("USA","DNK"),("USA","BEL"),
        ("USA","COL"),("USA","MEX"),("USA","BRA"),("USA","SAU"),("USA","ARE"),
        ("CHN","RUS"),("CHN","PAK"),("CHN","KAZ"),("CHN","MNG"),("CHN","IRN"),
        ("CHN","PRK"),("CHN","VNM"),("CHN","THA"),("CHN","IDN"),("CHN","SAU"),
        ("RUS","BLR"),("RUS","KAZ"),("RUS","ARM"),("RUS","IRN"),("RUS","SYR"),
        ("DEU","FRA"),("DEU","NLD"),("DEU","AUT"),("DEU","POL"),("GBR","AUS"),
        ("GBR","CAN"),("GBR","IRL"),("FRA","ESP"),("FRA","ITA"),("FRA","BEL"),
        ("JPN","KOR"),("JPN","AUS"),("KOR","AUS"),("IND","RUS"),("IND","ISR"),
        ("BRA","ARG"),("BRA","COL"),("ARE","SAU"),("ARE","QAT"),("EGY","SAU"),
        ("NGA","ZAF"),("KEN","ETH"),("GHA","NGA"),
    }
    date_range = (end - start).days + 1
    records = []
    for _ in range(n_events):
        c1, c2 = random.sample(countries, 2)
        pair = (c1, c2)
        if pair in tensions or (c2,c1) in tensions:
            tone = random.gauss(-3.5, 2.5)
            quad = random.choice([3, 3, 3, 4, 4])
        elif pair in alliances or (c2,c1) in alliances:
            tone = random.gauss(3.0, 2.0)
            quad = random.choice([1, 1, 2, 2])
        else:
            tone = random.gauss(0.0, 3.0)
            quad = random.choice([1, 2, 3, 4])
        event_code = random.choice(list(EVENT_ROOT_MAP.keys()))
        event_date = start + timedelta(days=random.randint(0, date_range - 1))
        records.append({
            "Actor1CountryCode": c1,
            "Actor2CountryCode": c2,
            "EventRootCode":     event_code,
            "QuadClass":         quad,
            "GoldsteinScale":    str(round(random.uniform(-10, 10), 1)),
            "AvgTone":           str(round(tone, 2)),
            "NumMentions":       str(random.randint(1, 200)),
            "NumArticles":       str(random.randint(1, 100)),
            "date":              event_date.strftime("%Y-%m-%d"),
            "SOURCEURL":         f"https://mock-news.example.com/{random.randint(10000,99999)}",
        })
    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} mock events across {date_range} days.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw GDELT or mock data for graph construction."""
    df = df.copy()

    # Normalise column names (handle any minor variants)
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "")
        if cl == "actor1countrycode":
            rename[c] = "Actor1CountryCode"
        elif cl == "actor2countrycode":
            rename[c] = "Actor2CountryCode"
    df = df.rename(columns=rename)

    # Ensure all required columns exist
    for col in ["Actor1CountryCode","Actor2CountryCode","AvgTone",
                "QuadClass","EventRootCode","date","NumMentions","GoldsteinScale"]:
        if col not in df.columns:
            df[col] = None

    # Filter to valid 3-letter ISO bilateral pairs
    df["Actor1CountryCode"] = df["Actor1CountryCode"].fillna("").str.strip()
    df["Actor2CountryCode"] = df["Actor2CountryCode"].fillna("").str.strip()
    df = df[df["Actor1CountryCode"].str.len() == 3]
    df = df[df["Actor2CountryCode"].str.len() == 3]
    df = df[df["Actor1CountryCode"] != ""]
    df = df[df["Actor2CountryCode"] != ""]
    df = df[df["Actor1CountryCode"] != df["Actor2CountryCode"]]

    if len(df) == 0:
        print("WARNING: No valid bilateral events after filtering.")
        return df.reset_index(drop=True)

    # Numeric conversions
    df["AvgTone"]        = pd.to_numeric(df["AvgTone"],        errors="coerce").fillna(0.0)
    df["QuadClass"]      = pd.to_numeric(df["QuadClass"],      errors="coerce").fillna(1).astype(int)
    df["NumMentions"]    = pd.to_numeric(df["NumMentions"],    errors="coerce").fillna(1)
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0.0)

    # Tone normalised to [-1, 1]
    df["tone_norm"] = df["AvgTone"].clip(-20, 20) / 20.0

    # Event root code: normalise to zero-padded 2-char string
    df["EventRootCode"] = (
        df["EventRootCode"].fillna("01").astype(str).str.strip()
         .apply(lambda x: x.zfill(2) if x.isdigit() else x[:2])
    )
    df["event_label"] = df["EventRootCode"].map(EVENT_ROOT_MAP).fillna("Unknown")

    def classify_type(row):
        code = str(row["EventRootCode"])[:2]
        qc   = int(row["QuadClass"])
        if code in ["06","07"]:        return "Trade/Aid"
        if code in ["18","19","20"]:   return "Military/Conflict"
        if qc in [1, 2]:              return "Cooperation"
        if qc in [3, 4]:              return "Conflict"
        return "Diplomatic"

    df["event_type"] = df.apply(classify_type, axis=1)

    print(f"Preprocessed: {len(df):,} valid events | "
          f"{df['Actor1CountryCode'].nunique()} source countries | "
          f"{df['Actor2CountryCode'].nunique()} target countries")
    return df.reset_index(drop=True)
