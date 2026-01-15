"""
Data loading functions for Google Stock ML project.
Handles price data, earnings, macro indicators, and EU break close flags.
Saves raw data to disk.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import ensure_dir, save_pickle, save_json


def load_price_data(
    tickers: List[str],
    start: str,
    end: str,
    base_ticker: str = "GOOGL"
) -> tuple:
    """
    Load price data for multiple tickers using NASDAQ calendar as master index.
    
    Returns:
        tuple: (prices_all DataFrame, data_dict with raw ticker data)
    """
    import pandas_market_calendars as mcal
    
    # 1. Define official calendar (NASDAQ) to avoid relying on Yahoo alone
    nyse = mcal.get_calendar('NASDAQ')
    valid_days = nyse.valid_days(start_date=start, end_date=end)
    master_index = pd.Index(valid_days.tz_localize(None).normalize(), name="Date")
    
    data_dict = {}
    
    for t in tickers:
        print(f"  [DOWNLOAD] {t}...")
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            print(f"  [WARN] No data for {t}")
            continue
        
        df.columns = [f"{t}_{c[0] if isinstance(c, tuple) else c}" for c in df.columns]
        
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        
        data_dict[t] = df
    
    if base_ticker not in data_dict:
        raise ValueError(f"{base_ticker} data is missing.")
    
    # 2. Build Master Table aligned to official calendar
    prices_all = pd.DataFrame(index=master_index)
    
    for t, df in data_dict.items():
        # Left join to calendar ensures we don't miss official trading days
        prices_all = prices_all.join(df, how="left")
    
    # Forward fill prices only
    price_cols = [c for c in prices_all.columns if any(s in c for s in ['_Open', '_High', '_Low', '_Close'])]
    prices_all[price_cols] = prices_all.sort_index()[price_cols].ffill()
    
    return prices_all, data_dict


def load_earnings_data(
    ticker: str = "GOOGL",
    prices_index: pd.DatetimeIndex = None,
    limit: int = 100
) -> pd.DataFrame:
    """Load earnings data for a ticker."""
    
    if prices_index is None:
        raise ValueError("prices_index is required")
    
    # Initialize empty earnings DataFrame
    earnings = pd.DataFrame(index=prices_index)
    earnings["is_earnings_day"] = 0
    earnings["eps_surprise_pct_yahoo"] = np.nan
    earnings["has_eps_surprise_yahoo"] = 0
    earnings["eps_surprise_pct_calc"] = np.nan
    earnings["has_eps_surprise_calc"] = 0
    
    # Try to load earnings data from yfinance
    try:
        tkr = yf.Ticker(ticker)
        edf = tkr.get_earnings_dates(limit=limit)
    except Exception as e:
        print(f"[WARN] Could not load earnings data for {ticker}: {e}")
        return earnings
    
    if edf is None or len(edf) == 0:
        print(f"[WARN] No earnings data returned for {ticker}")
        return earnings
    
    try:
        edf = edf.copy()
        idx = pd.to_datetime(edf.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = idx.normalize()
        
        edf.index = idx
        edf = edf[~edf.index.duplicated(keep="last")].sort_index()
    
        cols_lower = {c.lower(): c for c in edf.columns}
        
        def pick_col(possible_names):
            for name in possible_names:
                key = name.lower()
                if key in cols_lower:
                    return cols_lower[key]
            return None
        
        col_exp = pick_col(["EPS Estimate", "eps estimate", "Eps Estimate"])
        col_act = pick_col(["Reported EPS", "reported eps", "EPS Actual", "eps actual"])
        col_pct = pick_col(["Surprise(%)", "surprise(%)", "Surprise (%)", "surprise (%)"])
        
        eps_daily = pd.DataFrame(index=edf.index)
        eps_daily["eps_expected"] = edf[col_exp] if col_exp else np.nan
        eps_daily["eps_actual"] = edf[col_act] if col_act else np.nan
        
        if col_exp and col_act:
            eps_surprise = eps_daily["eps_actual"] - eps_daily["eps_expected"]
            denom = eps_daily["eps_expected"].abs()
            eps_daily["eps_surprise_pct_calc"] = np.where(
                denom > 0, 100.0 * (eps_surprise / denom), np.nan
            )
        else:
            eps_daily["eps_surprise_pct_calc"] = np.nan
        
        eps_daily["eps_surprise_pct_yahoo"] = edf[col_pct] if col_pct else np.nan
        
        # Align to official trading days (vectorized - as in notebook)
        eps_on_trading_days = eps_daily.reindex(prices_index)
        
        earnings["is_earnings_day"] = prices_index.isin(eps_daily.index).astype("int8")
        earnings["eps_surprise_pct_yahoo"] = eps_on_trading_days["eps_surprise_pct_yahoo"].values
        earnings["eps_surprise_pct_calc"] = eps_on_trading_days["eps_surprise_pct_calc"].values
        earnings["has_eps_surprise_yahoo"] = earnings["eps_surprise_pct_yahoo"].notna().astype("int8")
        earnings["has_eps_surprise_calc"] = earnings["eps_surprise_pct_calc"].notna().astype("int8")
    
    except Exception as e:
        print(f"[WARN] Error processing earnings data for {ticker}: {e}")
    
    return earnings


def build_eu_info_gap_flags(
    us_index: pd.DatetimeIndex,
    data_dict: Dict[str, pd.DataFrame],
    eu_ticker: str = "^GDAXI",
) -> pd.DataFrame:
    """
    Build EU "info gap" flags - identifies days when US was closed but EU was open.
    
    Returns:
        - EU_break_close_flag (int8): 1 if US was closed previously while EU was open.
        - EU_break_close_up   (int8): 1 if EU cumulative return during US holiday was positive.
        - EU_break_close_down (int8): 1 if EU cumulative return during US holiday was negative.
    
    Args:
        us_index: US trading days index (from prices_all)
        data_dict: Dictionary with ticker data (must include eu_ticker)
        eu_ticker: EU ticker symbol (default ^GDAXI)
    """
    # 1. Get EU data from data_dict that was already downloaded
    if eu_ticker not in data_dict:
        print(f"[WARN] {eu_ticker} missing from data_dict, returning empty flags")
        out = pd.DataFrame(index=us_index)
        out["EU_break_close_flag"] = np.int8(0)
        out["EU_break_close_up"] = np.int8(0)
        out["EU_break_close_down"] = np.int8(0)
        return out
    
    eu_data = data_dict[eu_ticker].copy()
    eu_days = eu_data.index.normalize()
    us_days = us_index.normalize()
    
    # 2. Identify gap days: Europe open, US closed
    gap_days = eu_days.difference(us_days)
    
    # 3. Compute log returns (allows summing over consecutive holiday days)
    close_col = f"{eu_ticker}_Close"
    if close_col not in eu_data.columns:
        print(f"[WARN] {close_col} not found in EU data, returning empty flags")
        out = pd.DataFrame(index=us_index)
        out["EU_break_close_flag"] = np.int8(0)
        out["EU_break_close_up"] = np.int8(0)
        out["EU_break_close_down"] = np.int8(0)
        return out
    
    eu_log_ret = np.log(eu_data[close_col] / eu_data[close_col].shift(1))
    
    # 4. Create events table for gap days
    eu_gap_events = pd.DataFrame(index=gap_days)
    eu_gap_events["gap_return"] = eu_log_ret.reindex(gap_days)
    
    # 5. Map to next US trading day
    pos = np.searchsorted(us_days, eu_gap_events.index, side="left")
    valid_mask = pos < len(us_days)
    
    eu_gap_events['target_us_date'] = pd.NaT  # NaT for datetime instead of np.nan
    eu_gap_events.loc[valid_mask, 'target_us_date'] = us_days[pos[valid_mask]]
    
    # 6. Aggregate (for long holidays, sum all EU returns to US opening day)
    agg_gap = eu_gap_events.dropna(subset=['target_us_date']).groupby('target_us_date')["gap_return"].sum()
    
    # 7. Create output table
    out = pd.DataFrame(index=us_days)
    out["EU_break_close_flag"] = np.int8(0)
    out["EU_break_close_up"] = np.int8(0)
    out["EU_break_close_down"] = np.int8(0)
    
    out.loc[agg_gap.index, "EU_break_close_flag"] = 1
    out.loc[agg_gap.index, "EU_break_close_up"] = (agg_gap > 0).astype("int8")
    out.loc[agg_gap.index, "EU_break_close_down"] = (agg_gap < 0).astype("int8")
    
    # Restore original index
    out.index = us_index
    
    for col in ["EU_break_close_flag", "EU_break_close_up", "EU_break_close_down"]:
        out[col] = out[col].astype("int8")
    
    return out


# Keep old function for backward compatibility (deprecated)
def build_eu_break_close_flags(
    start: str,
    end: str,
    googl_index: pd.DatetimeIndex,
    eu_ticker: str = "^GDAXI",
    gap_days_threshold: int = 2,
    apply_to: str = "next_us_trading_day",
) -> pd.DataFrame:
    """
    DEPRECATED: Use build_eu_info_gap_flags instead.
    This function downloads data separately - inefficient.
    """
    print("[WARN] build_eu_break_close_flags is deprecated, use build_eu_info_gap_flags")
    print(f"  [DOWNLOAD] {eu_ticker} for EU break close...")
    eu = yf.download(eu_ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if eu is None or eu.empty:
        print(f"  [WARN] No data for {eu_ticker}, returning empty flags")
        out = pd.DataFrame(index=googl_index)
        out["EU_break_close_flag"] = np.int8(0)
        out["EU_break_close_up"] = np.int8(0)
        out["EU_break_close_down"] = np.int8(0)
        return out
    
    eu.columns = [f"{eu_ticker}_{c[0] if isinstance(c, tuple) else c}" for c in eu.columns]
    
    if isinstance(eu.index, pd.DatetimeIndex) and eu.index.tz is not None:
        eu.index = eu.index.tz_convert(None)
    
    eu = eu.sort_index()
    close_col = f"{eu_ticker}_Close"
    
    eu_close = eu[close_col].copy()
    eu_dates = eu_close.index.normalize()
    
    next_dates = eu_dates.to_series().shift(-1)
    delta_days = (next_dates - eu_dates.to_series()).dt.days
    is_break_close = (delta_days >= gap_days_threshold).fillna(False)
    
    eu_ret_1d = eu_close.pct_change()
    break_ret_1d = eu_ret_1d.where(is_break_close.values, np.nan)
    break_dir = np.sign(break_ret_1d)
    
    eu_event = pd.DataFrame({
        "EU_break_close_flag": is_break_close.astype("int8").values,
        "EU_break_close_up": (break_dir > 0).astype("int8").values,
        "EU_break_close_down": (break_dir < 0).astype("int8").values,
    }, index=pd.DatetimeIndex(eu_dates.values)).sort_index()
    
    eu_event = eu_event[eu_event["EU_break_close_flag"] == 1].copy()
    
    g_idx = pd.DatetimeIndex(googl_index.normalize()).sort_values()
    out = pd.DataFrame(index=g_idx)
    out["EU_break_close_flag"] = np.int8(0)
    out["EU_break_close_up"] = np.int8(0)
    out["EU_break_close_down"] = np.int8(0)
    
    if eu_event.empty:
        return out
    
    eu_event_dates = eu_event.index.values
    
    if apply_to == "next_us_trading_day":
        pos = np.searchsorted(g_idx.values, eu_event_dates, side="left")
        valid = pos < len(g_idx)
        mapped_dates = g_idx.values[pos[valid]]
        src_dates = eu_event_dates[valid]
        out.loc[mapped_dates, eu_event.columns] = eu_event.loc[src_dates, eu_event.columns].values
    else:
        common = np.intersect1d(g_idx.values, eu_event_dates)
        if len(common) > 0:
            out.loc[common, eu_event.columns] = eu_event.loc[common, eu_event.columns].values
    
    for col in ["EU_break_close_flag", "EU_break_close_up", "EU_break_close_down"]:
        out[col] = out[col].astype("int8")
    
    return out


def load_macro_data(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load macro data from FRED.
    
    Creates features:
    - CPI_pct_mom: CPI month-over-month percent change
    - CPI_accel_pct_mom: Acceleration of CPI change
    - FEDFUNDS_delta_mom: Fed funds rate change
    - FEDFUNDS_changed: Binary flag if rate changed
    - FEDFUNDS_level: Fed funds rate level
    - *_is_missing: Missingness flags
    - CPI_release_day: Binary flag for CPI release days
    - FEDFUNDS_release_day: Binary flag for rate release days
    """
    try:
        import pandas_datareader.data as pdr
    except ImportError:
        print("[WARN] pandas_datareader not installed, skipping macro data")
        return pd.DataFrame(index=index)
    
    start = index.min().strftime("%Y-%m-%d")
    end = index.max().strftime("%Y-%m-%d")
    
    # --- Pull monthly series from FRED ---
    try:
        cpi = pdr.DataReader("CPIAUCSL", "fred", start, end).rename(columns={"CPIAUCSL": "CPI"})
        cpi.index = pd.to_datetime(cpi.index)
        
        # CPI features (computed on monthly data BEFORE ffill)
        cpi["CPI_pct_mom"] = cpi["CPI"].pct_change(1, fill_method=None)
        cpi["CPI_accel_pct_mom"] = cpi["CPI_pct_mom"] - cpi["CPI_pct_mom"].shift(1)
        cpi_feats_monthly = cpi[["CPI_pct_mom", "CPI_accel_pct_mom"]].copy()
    except Exception as e:
        print(f"[WARN] Could not load CPI: {e}")
        cpi_feats_monthly = pd.DataFrame()
    
    try:
        rate = pdr.DataReader("FEDFUNDS", "fred", start, end).rename(columns={"FEDFUNDS": "FEDFUNDS"})
        rate.index = pd.to_datetime(rate.index)
        
        # FEDFUNDS features (computed on monthly data BEFORE ffill)
        rate["FEDFUNDS_delta_mom"] = rate["FEDFUNDS"].diff(1)
        rate["FEDFUNDS_changed"] = (rate["FEDFUNDS_delta_mom"].fillna(0) != 0).astype("int8")
        rate["FEDFUNDS_level"] = rate["FEDFUNDS"].copy()
        rate_feats_monthly = rate[["FEDFUNDS_delta_mom", "FEDFUNDS_changed", "FEDFUNDS_level"]].copy()
    except Exception as e:
        print(f"[WARN] Could not load FEDFUNDS: {e}")
        rate_feats_monthly = pd.DataFrame()
    
    # Upsample to daily and forward-fill
    if not cpi_feats_monthly.empty:
        cpi_feats_daily = cpi_feats_monthly.resample("D").ffill()
    else:
        cpi_feats_daily = pd.DataFrame()
    
    if not rate_feats_monthly.empty:
        rate_feats_daily = rate_feats_monthly.resample("D").ffill()
    else:
        rate_feats_daily = pd.DataFrame()
    
    # Combine
    if not cpi_feats_daily.empty and not rate_feats_daily.empty:
        macro_daily = pd.concat([cpi_feats_daily, rate_feats_daily], axis=1)
    elif not cpi_feats_daily.empty:
        macro_daily = cpi_feats_daily
    elif not rate_feats_daily.empty:
        macro_daily = rate_feats_daily
    else:
        return pd.DataFrame(index=index)
    
    # Align to trading dates index
    macro_daily.index = pd.to_datetime(macro_daily.index)
    macro_aligned = macro_daily.reindex(index)
    
    # --- Missingness flags BEFORE ffill ---
    FLAG_SUFFIX = "_is_missing"
    macro_numeric_to_fill0 = ["CPI_pct_mom", "CPI_accel_pct_mom", "FEDFUNDS_delta_mom"]
    
    for col in macro_numeric_to_fill0:
        if col in macro_aligned.columns:
            macro_aligned[f"{col}{FLAG_SUFFIX}"] = macro_aligned[col].isna().astype("int8")
    
    # FFill macro gaps
    macro_cols = macro_aligned.columns.tolist()
    macro_aligned[macro_cols] = macro_aligned[macro_cols].ffill()
    
    # Fill leading NaNs with 0 for numeric cols
    for col in macro_numeric_to_fill0:
        if col in macro_aligned.columns:
            macro_aligned[col] = macro_aligned[col].fillna(0.0)
    
    # Enforce FEDFUNDS_changed to stay binary int8
    if "FEDFUNDS_changed" in macro_aligned.columns:
        macro_aligned["FEDFUNDS_changed"] = (
            macro_aligned["FEDFUNDS_changed"]
            .fillna(0)
            .clip(0, 1)
            .astype("int8")
        )
    
    # --- Release-day flags ---
    if "CPI_pct_mom" in macro_aligned.columns:
        cpi_series = macro_aligned["CPI_pct_mom"].astype("float64")
        macro_aligned["CPI_release_day"] = (
            cpi_series.notna() & cpi_series.ne(cpi_series.shift(1))
        ).astype("int8")
    
    if "FEDFUNDS_level" in macro_aligned.columns:
        ff_series = macro_aligned["FEDFUNDS_level"].astype("float64")
        macro_aligned["FEDFUNDS_release_day"] = (
            ff_series.notna() & ff_series.ne(ff_series.shift(1))
        ).astype("int8")
    
    # Enforce all flags to int8
    for col in macro_aligned.columns:
        if col.endswith(FLAG_SUFFIX) or col.endswith("_release_day"):
            macro_aligned[col] = macro_aligned[col].fillna(0).clip(0, 1).astype("int8")
    
    return macro_aligned


def load_all_data(
    price_tickers: List[str],
    start: str,
    end: str,
    base_ticker: str = "GOOGL",
    eu_config: Optional[Dict] = None,
    load_macro: bool = True,
    paths: Optional[Dict[str, Path]] = None
) -> pd.DataFrame:
    """
    Load all data sources and combine.
    
    Args:
        price_tickers: List of price tickers
        start: Start date
        end: End date
        base_ticker: Base ticker for timeline
        eu_config: EU break close configuration
        load_macro: Whether to load macro data
        paths: Optional paths dict with key "raw" for saving
    
    Returns:
        Combined DataFrame
    """
    # Ensure EU ticker is in price_tickers if EU is enabled
    eu_ticker = "^GDAXI"
    if eu_config and eu_config.get("enabled", False):
        eu_ticker = eu_config.get("eu_ticker", "^GDAXI")
        if eu_ticker not in price_tickers:
            price_tickers = price_tickers + [eu_ticker]
    
    print(f"[INFO] Loading price data for {len(price_tickers)} tickers...")
    full_df, data_dict = load_price_data(price_tickers, start, end, base_ticker)
    print(f"[INFO] Price data shape: {full_df.shape}")
    
    print("[INFO] Loading earnings data...")
    earnings = load_earnings_data(base_ticker, full_df.index)
    full_df = full_df.join(earnings, how="left")
    
    if eu_config and eu_config.get("enabled", False):
        print("[INFO] Building EU break close flags...")
        eu_flags = build_eu_info_gap_flags(
            us_index=full_df.index,
            data_dict=data_dict,
            eu_ticker=eu_ticker,
        )
        
        # Remove existing columns if function runs again
        cols_to_drop = [c for c in eu_flags.columns if c in full_df.columns]
        if cols_to_drop:
            full_df = full_df.drop(columns=cols_to_drop)
        
        full_df = full_df.join(eu_flags, how="left")
        
        for col in ["EU_break_close_flag", "EU_break_close_up", "EU_break_close_down"]:
            if col in full_df.columns:
                full_df[col] = full_df[col].fillna(0).astype("int8")
        
        print(f"[INFO] EU break close events: {full_df['EU_break_close_flag'].sum()}")
    
    if load_macro:
        print("[INFO] Loading macro data...")
        macro = load_macro_data(full_df.index)
        full_df = full_df.join(macro, how="left")
    
    print(f"[INFO] Final raw data shape: {full_df.shape}")
    
    # === SAVE RAW DATA ===
    if paths and "raw" in paths:
        raw_dir = ensure_dir(Path(paths["raw"]))
        out_path = raw_dir / "prices_raw.pkl"
        full_df.to_pickle(out_path)
        print(f"[INFO] Saved raw data to {out_path}")
        
        # Save metadata
        meta = {
            "tickers": price_tickers,
            "start": start,
            "end": end,
            "base_ticker": base_ticker,
            "shape": list(full_df.shape),
            "date_min": str(full_df.index.min()),
            "date_max": str(full_df.index.max()),
        }
        save_json(meta, raw_dir / "raw_data_meta.json")
    
    return full_df
