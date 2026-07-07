"""
World Bank World Development Indicators (WDI) client.

Implements the data pull described in the paper's section 3.2: one country per
Sub-Saharan sub-region, filtered to the "Individuals using the Internet",
"Borrowing from a financial institution", and "Health services" indicator families.

This hits the public WDI REST API directly — no API key required:
    https://api.worldbank.org/v2/country/{iso2}/indicator/{indicator}?format=json

Requires outbound internet access to api.worldbank.org. If that's unavailable in your
environment (e.g. a sandboxed CI runner), `fetch_all_indicators` will raise a clear
RuntimeError per failed call rather than silently returning empty data — catch it and
fall back to a cached CSV in `data/raw/` if needed.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from dat_framework.config import SUB_SAHARAN_COUNTRIES, WORLD_BANK_INDICATORS

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# Reused across calls so repeated requests to api.worldbank.org keep the same
# TCP/TLS connection alive (connection pooling) instead of each call paying a
# fresh DNS-lookup + handshake cost — on Windows in particular, doing that 9+
# times in a row (one per country/indicator pair) can add up to something
# that looks like a hang even though no single call is actually broken.
_SESSION = requests.Session()
# Ignore any HTTP_PROXY/HTTPS_PROXY/NO_PROXY inherited from the system
# environment. A stale or misconfigured proxy is one of the most common
# causes of "works fine when I test it standalone, hangs every time from
# inside the app" — trust_env=False makes requests talk to the internet
# directly regardless of what's set system-wide.
_SESSION.trust_env = False


def fetch_indicator(
    country_iso2: str,
    indicator_code: str,
    start_year: int = 2010,
    end_year: int = 2023,
    per_page: int = 200,
    timeout: int = 10,
    max_retries: int = 2,
) -> pd.DataFrame:
    """Fetch a single WDI indicator for a single country.

    Returns a tidy DataFrame with columns: country, country_iso2, indicator_code,
    indicator_name, year, value.
    """
    url = BASE_URL.format(country=country_iso2, indicator=indicator_code)
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": per_page,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
            last_err = exc
            if attempt < max_retries:
                time.sleep(1.0 * attempt)
    else:
        raise RuntimeError(
            f"Failed to fetch {indicator_code} for {country_iso2} after "
            f"{max_retries} attempts: {last_err}"
        )

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        # World Bank returns [metadata, null] when there's no data for the query.
        return pd.DataFrame(
            columns=[
                "country", "country_iso2", "indicator_code",
                "indicator_name", "year", "value",
            ]
        )

    records = payload[1]
    rows = [
        {
            "country": r["country"]["value"],
            "country_iso2": country_iso2,
            "indicator_code": indicator_code,
            "indicator_name": r["indicator"]["value"],
            "year": int(r["date"]),
            "value": r["value"],
        }
        for r in records
    ]
    return pd.DataFrame(rows)


def fetch_all_indicators(
    countries: Optional[Dict[str, str]] = None,
    indicators: Optional[Dict[str, str]] = None,
    start_year: int = 2010,
    end_year: int = 2023,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Fetch every configured indicator for every configured country.

    Defaults to `config.SUB_SAHARAN_COUNTRIES` and `config.WORLD_BANK_INDICATORS`.
    `progress_callback(done, total, label)` is called after each call completes
    (success or failure) so a caller — e.g. the Streamlit app — can show real
    progress instead of a single opaque spinner for the whole batch.
    """
    countries = countries or SUB_SAHARAN_COUNTRIES
    indicators = indicators or WORLD_BANK_INDICATORS

    calls = [(iso2, code) for iso2 in countries for code in indicators]
    frames: List[pd.DataFrame] = []
    errors: List[str] = []

    for i, (iso2, code) in enumerate(calls, start=1):
        label = f"{countries[iso2]} — {indicators[code]}"
        try:
            df = fetch_indicator(iso2, code, start_year, end_year)
            frames.append(df)
        except RuntimeError as exc:
            errors.append(str(exc))
        if progress_callback is not None:
            progress_callback(i, len(calls), label)
        if i < len(calls):
            time.sleep(0.3)  # small gap between calls to avoid bursty rate-limiting

    if errors and not frames:
        raise RuntimeError(
            "All World Bank API calls failed — check internet access to "
            "api.worldbank.org. Errors:\n" + "\n".join(errors)
        )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if errors:
        # Partial failure: surface a warning but still return what we got.
        print(f"[worldbank] {len(errors)} indicator/country calls failed:")
        for e in errors:
            print(f"  - {e}")
    return combined


def latest_value_per_country(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long panel to the most recent non-null value per country/indicator."""
    clean = df.dropna(subset=["value"])
    if clean.empty:
        return clean
    idx = clean.groupby(["country_iso2", "indicator_code"])["year"].idxmax()
    return clean.loc[idx].reset_index(drop=True)


if __name__ == "__main__":
    data = fetch_all_indicators()
    out_path = "data/raw/worldbank_wdi.csv"
    data.to_csv(out_path, index=False)
    print(f"Saved {len(data)} rows to {out_path}")
