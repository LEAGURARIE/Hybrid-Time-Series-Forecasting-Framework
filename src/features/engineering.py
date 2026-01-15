"""
Feature engineering for Google Stock ML project.
Matches notebook cells 18, 20, 22, 24, 26, 28 exactly.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

EPS = 1e-9  # Default epsilon for numerical safety


# ==============================================================================
# CELL 18: RAW TRANSFORMATIONS
# ==============================================================================

def add_raw_transformations(
    df: pd.DataFrame,
    exclude_raw_ohlc: Optional[List[str]] = None,
    eps: float = EPS
) -> pd.DataFrame:
    """
    Add raw price transformations (log returns, abs returns, ratios).
    Matches notebook Cell 18 (BLOCK 6).
    
    Creates per ticker:
    - logret_cc, logret_oc, logret_gap_co
    - abs_logret_cc, abs_logret_oc, abs_logret_gap_co
    - log_hl, close_pos_hl, close_pos_hl_centered
    - logret_cc_lag1, logret_cc_lag5, logret_cc_lag21
    - log_vol, log_vol_chg_1d, log_dollar_vol (if Volume exists)
    """
    df = df.copy()
    exclude_raw_ohlc = set(exclude_raw_ohlc or [])
    
    # Identify OHLCV ticker prefixes
    def has_cols(prefix: str, required: list) -> bool:
        return all(f"{prefix}_{c}" in df.columns for c in required)
    
    required_ohlc = ["Open", "High", "Low", "Close"]
    required_ohlcv = ["Open", "High", "Low", "Close", "Volume"]
    
    prefixes = sorted({c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_Close")})
    
    ohlc_prefixes = [p for p in prefixes if has_cols(p, required_ohlc) and p not in exclude_raw_ohlc]
    ohlcv_prefixes = [p for p in prefixes if has_cols(p, required_ohlcv) and p not in exclude_raw_ohlc]
    
    print(f"[INFO] Raw transforms for {len(ohlc_prefixes)} OHLC tickers (excluded {sorted(exclude_raw_ohlc)})")
    
    new_cols = {}
    
    # Raw transforms per ticker (NO rolling)
    for p in ohlc_prefixes:
        o = df[f"{p}_Open"].astype("float64")
        h = df[f"{p}_High"].astype("float64")
        l = df[f"{p}_Low"].astype("float64")
        c = df[f"{p}_Close"].astype("float64")
        
        # log returns
        new_cols[f"{p}_logret_cc"] = np.log((c + eps) / (c.shift(1) + eps))
        new_cols[f"{p}_logret_oc"] = np.log((c + eps) / (o + eps))
        new_cols[f"{p}_logret_gap_co"] = np.log((o + eps) / (c.shift(1) + eps))
        
        # abs variants
        new_cols[f"{p}_abs_logret_cc"] = new_cols[f"{p}_logret_cc"].abs()
        new_cols[f"{p}_abs_logret_oc"] = new_cols[f"{p}_logret_oc"].abs()
        new_cols[f"{p}_abs_logret_gap_co"] = new_cols[f"{p}_logret_gap_co"].abs()
        
        # intraday range
        new_cols[f"{p}_log_hl"] = np.log((h + eps) / (l + eps))
        
        # close position within High-Low range
        denom_hl = (h - l).replace(0.0, np.nan)
        close_pos = (c - l) / denom_hl
        new_cols[f"{p}_close_pos_hl"] = close_pos
        new_cols[f"{p}_close_pos_hl_centered"] = close_pos - 0.5
        
        # lags for logret_cc
        for lag in [1, 5, 21]:
            new_cols[f"{p}_logret_cc_lag{lag}"] = new_cols[f"{p}_logret_cc"].shift(lag)
    
    # Volume features (only where Volume exists)
    for p in ohlcv_prefixes:
        v = df[f"{p}_Volume"].astype("float64")
        c = df[f"{p}_Close"].astype("float64")
        
        new_cols[f"{p}_log_vol"] = np.log(v + 1.0)
        new_cols[f"{p}_log_vol_chg_1d"] = new_cols[f"{p}_log_vol"] - new_cols[f"{p}_log_vol"].shift(1)
        
        dollar_vol = (c * v).astype("float64")
        new_cols[f"{p}_log_dollar_vol"] = np.log(dollar_vol + 1.0)
    
    # Attach to df
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} raw feature columns")
    return df


# ==============================================================================
# CELL 20: ROLLING STATISTICS
# ==============================================================================

def add_rolling_statistics(
    df: pd.DataFrame,
    w_short: int = 5,
    w_long: int = 21,
    do_volume_rolling: bool = True,
    eps: float = EPS
) -> pd.DataFrame:
    """
    Add rolling statistics for returns, abs returns, HL range, volume.
    Matches notebook Cell 20 (BLOCK 7).
    
    Creates per ticker:
    - logret_cc_mean_{w}, logret_cc_std_{w}, logret_cc_z_{w_long}
    - abs_logret_cc_mean_{w}, abs_logret_cc_std_{w}
    - log_hl_mean_{w}, log_hl_std_{w}, log_hl_z_{w_long}
    - log_vol_mean_{w}, log_vol_std_{w}, log_vol_z_{w_long} (if enabled)
    """
    df = df.copy()
    
    # Determine tickers that have raw series
    tickers = sorted({
        c.replace("_logret_cc", "")
        for c in df.columns
        if c.endswith("_logret_cc") and not c.endswith("_abs_logret_cc")
    })
    
    print(f"[INFO] Rolling stats for {len(tickers)} tickers (w_short={w_short}, w_long={w_long})")
    
    new_cols = {}
    
    for p in tickers:
        col_r = f"{p}_logret_cc"
        col_ar = f"{p}_abs_logret_cc"
        col_hl = f"{p}_log_hl"
        
        if col_r not in df.columns:
            continue
        
        r = df[col_r].astype("float64")
        
        # rolling mean/std of logret_cc
        r_m_s = r.rolling(w_short, min_periods=w_short).mean()
        r_s_s = r.rolling(w_short, min_periods=w_short).std()
        r_m_l = r.rolling(w_long, min_periods=w_long).mean()
        r_s_l = r.rolling(w_long, min_periods=w_long).std()
        
        new_cols[f"{p}_logret_cc_mean_{w_short}"] = r_m_s
        new_cols[f"{p}_logret_cc_std_{w_short}"] = r_s_s
        new_cols[f"{p}_logret_cc_mean_{w_long}"] = r_m_l
        new_cols[f"{p}_logret_cc_std_{w_long}"] = r_s_l
        
        # z-score vs long window
        new_cols[f"{p}_logret_cc_z_{w_long}"] = (r - r_m_l) / (r_s_l + eps)
        
        # abs_logret_cc mean/std
        if col_ar in df.columns:
            ar = df[col_ar].astype("float64")
            new_cols[f"{p}_abs_logret_cc_mean_{w_short}"] = ar.rolling(w_short, min_periods=w_short).mean()
            new_cols[f"{p}_abs_logret_cc_std_{w_short}"] = ar.rolling(w_short, min_periods=w_short).std()
            new_cols[f"{p}_abs_logret_cc_mean_{w_long}"] = ar.rolling(w_long, min_periods=w_long).mean()
            new_cols[f"{p}_abs_logret_cc_std_{w_long}"] = ar.rolling(w_long, min_periods=w_long).std()
        
        # HL rolling mean/std + z-score
        if col_hl in df.columns:
            hl = df[col_hl].astype("float64")
            hl_m_s = hl.rolling(w_short, min_periods=w_short).mean()
            hl_s_s = hl.rolling(w_short, min_periods=w_short).std()
            hl_m_l = hl.rolling(w_long, min_periods=w_long).mean()
            hl_s_l = hl.rolling(w_long, min_periods=w_long).std()
            
            new_cols[f"{p}_log_hl_mean_{w_short}"] = hl_m_s
            new_cols[f"{p}_log_hl_std_{w_short}"] = hl_s_s
            new_cols[f"{p}_log_hl_mean_{w_long}"] = hl_m_l
            new_cols[f"{p}_log_hl_std_{w_long}"] = hl_s_l
            new_cols[f"{p}_log_hl_z_{w_long}"] = (hl - hl_m_l) / (hl_s_l + eps)
    
    # Optional: Volume rolling stats
    if do_volume_rolling:
        for p in tickers:
            col_lv = f"{p}_log_vol"
            if col_lv not in df.columns:
                continue
            
            lv = df[col_lv].astype("float64")
            
            lv_m_s = lv.rolling(w_short, min_periods=w_short).mean()
            lv_s_s = lv.rolling(w_short, min_periods=w_short).std()
            lv_m_l = lv.rolling(w_long, min_periods=w_long).mean()
            lv_s_l = lv.rolling(w_long, min_periods=w_long).std()
            
            new_cols[f"{p}_log_vol_mean_{w_short}"] = lv_m_s
            new_cols[f"{p}_log_vol_std_{w_short}"] = lv_s_s
            new_cols[f"{p}_log_vol_mean_{w_long}"] = lv_m_l
            new_cols[f"{p}_log_vol_std_{w_long}"] = lv_s_l
            new_cols[f"{p}_log_vol_z_{w_long}"] = (lv - lv_m_l) / (lv_s_l + eps)
    
    # Attach
    roll_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, roll_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} rolling-stat columns")
    return df


# ==============================================================================
# CELL 22: CROSS ASSET FEATURES
# ==============================================================================

def add_cross_asset_features(
    df: pd.DataFrame,
    base: str = "GOOGL",
    peers: List[str] = None,
    windows: List[int] = None,
    eps: float = EPS
) -> pd.DataFrame:
    """
    Add rolling correlation and beta vs peers.
    Matches notebook Cell 22 (BLOCK 8).
    
    Creates:
    - {base}_corr_{peer}_{w}
    - {base}_beta_{peer}_{w}
    """
    df = df.copy()
    peers = peers or ["SPY", "QQQ", "MSFT", "NVDA"]
    windows = windows or [5, 21]
    
    base_col = f"{base}_logret_cc"
    if base_col not in df.columns:
        print(f"[WARN] Missing {base_col}, skipping cross-asset features")
        return df
    
    new_cols = {}
    r_base = df[base_col].astype("float64")
    
    for p in peers:
        peer_col = f"{p}_logret_cc"
        if peer_col not in df.columns:
            print(f"[WARN] Missing {peer_col}, skipping")
            continue
        
        r_peer = df[peer_col].astype("float64")
        
        for w in windows:
            w = int(w)
            new_cols[f"{base}_corr_{p}_{w}"] = r_base.rolling(w, min_periods=w).corr(r_peer)
            
            cov = r_base.rolling(w, min_periods=w).cov(r_peer)
            var = r_peer.rolling(w, min_periods=w).var()
            new_cols[f"{base}_beta_{p}_{w}"] = cov / (var + eps)
    
    # Attach
    cross_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, cross_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} cross-asset columns (base={base})")
    return df


# ==============================================================================
# CELL 24: REGIME / INTERACTION FEATURES
# ==============================================================================

def add_regime_features(
    df: pd.DataFrame,
    base: str = "GOOGL",
    w_short: int = 5,
    w_long: int = 21,
    peers: List[str] = None,
    eps: float = EPS
) -> pd.DataFrame:
    """
    Add volatility regime and interaction features.
    Matches notebook Cell 24 (BLOCK 9).
    
    Creates:
    - vol_ratio_{w_short}_{w_long}, vol_diff_{w_short}_{w_long}
    - vol_regime_score
    - price_struct_1, price_struct_2
    - ctx_beta_vol_{peer}, ctx_corr_volratio_{peer}
    - vix_vol_interact
    - fedfunds_trend_interact, cpi_trend_interact
    """
    df = df.copy()
    peers = peers or ["SPY", "QQQ", "MSFT", "NVDA"]
    
    new_cols = {}
    
    # Required columns
    std_s_col = f"{base}_logret_cc_std_{w_short}"
    std_l_col = f"{base}_logret_cc_std_{w_long}"
    z_ret_col = f"{base}_logret_cc_z_{w_long}"
    
    if std_s_col not in df.columns or std_l_col not in df.columns:
        print(f"[WARN] Missing volatility columns for {base}, skipping regime features")
        return df
    
    std_s = df[std_s_col].astype("float64")
    std_l = df[std_l_col].astype("float64")
    
    # Volatility ratio and diff
    new_cols[f"{base}_vol_ratio_{w_short}_{w_long}"] = std_s / (std_l + eps)
    new_cols[f"{base}_vol_diff_{w_short}_{w_long}"] = std_s - std_l
    
    # Vol regime score
    if z_ret_col in df.columns:
        z_ret = df[z_ret_col].astype("float64")
        new_cols[f"{base}_vol_regime_score"] = z_ret * (std_s / (std_l + eps))
    
    # Price Structure
    abs_ret_col = f"{base}_abs_logret_cc"
    abs_ret_mean_l_col = f"{base}_abs_logret_cc_mean_{w_long}"
    abs_ret_std_l_col = f"{base}_abs_logret_cc_std_{w_long}"
    close_pos_col = f"{base}_close_pos_hl"
    log_hl_z_col = f"{base}_log_hl_z_{w_long}"
    
    if all(c in df.columns for c in [abs_ret_col, abs_ret_mean_l_col, abs_ret_std_l_col, close_pos_col]):
        abs_ret = df[abs_ret_col].astype("float64")
        abs_ret_mean_l = df[abs_ret_mean_l_col].astype("float64")
        abs_ret_std_l = df[abs_ret_std_l_col].astype("float64")
        abs_ret_z = (abs_ret - abs_ret_mean_l) / (abs_ret_std_l + eps)
        
        new_cols[f"{base}_price_struct_1"] = df[close_pos_col].astype("float64") * abs_ret_z
    
    if log_hl_z_col in df.columns:
        new_cols[f"{base}_price_struct_2"] = (
            df[log_hl_z_col].astype("float64") * (std_s / (std_l + eps))
        )
    
    # Market Context (BASE vs peers)
    for p in peers:
        beta_col = f"{base}_beta_{p}_{w_long}"
        corr_col = f"{base}_corr_{p}_{w_long}"
        
        if beta_col in df.columns:
            new_cols[f"{base}_ctx_beta_vol_{p}"] = df[beta_col].astype("float64") * std_l
        
        if corr_col in df.columns:
            new_cols[f"{base}_ctx_corr_volratio_{p}"] = (
                df[corr_col].astype("float64") * (std_s / (std_l + eps))
            )
    
    # Macro context (event-aware)
    trend_l_col = f"{base}_logret_cc_mean_{w_long}"
    if trend_l_col in df.columns:
        trend_l = df[trend_l_col].astype("float64")
        
        # VIX level × volatility
        if "^VIX_Close" in df.columns:
            new_cols[f"{base}_vix_vol_interact"] = df["^VIX_Close"].astype("float64") * std_l
        
        # FEDFUNDS delta × trend, ONLY on release day
        if "FEDFUNDS_delta_mom" in df.columns and "FEDFUNDS_release_day" in df.columns:
            new_cols[f"{base}_fedfunds_trend_interact"] = (
                df["FEDFUNDS_delta_mom"].astype("float64") *
                df["FEDFUNDS_release_day"].astype("float64") *
                trend_l
            )
        
        # CPI pct_mom × trend, ONLY on release day
        if "CPI_pct_mom" in df.columns and "CPI_release_day" in df.columns:
            new_cols[f"{base}_cpi_trend_interact"] = (
                df["CPI_pct_mom"].astype("float64") *
                df["CPI_release_day"].astype("float64") *
                trend_l
            )
    
    # Attach
    inter_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, inter_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} regime/interaction columns")
    return df


# ==============================================================================
# CELL 26: VIX / TNX FEATURES
# ==============================================================================

def add_vix_tnx_features(
    df: pd.DataFrame,
    eps: float = EPS
) -> pd.DataFrame:
    """
    Add special VIX and TNX features.
    Matches notebook Cell 26 (BLOCK 10).
    
    Creates:
    - VIX_log_level, VIX_delta_1d, VIX_abs_delta_1d, VIX_log_hl, VIX_range_frac, VIX_gap_oc
    - VIX_delta_1d_lag{1,5,21}, VIX_abs_delta_1d_lag{1,5,21}
    - TNX_level, TNX_delta_1d, TNX_abs_delta_1d, TNX_delta_5d, TNX_log_hl, TNX_range, TNX_gap_oc
    - TNX_delta_1d_lag{1,5,21}, TNX_abs_delta_1d_lag{1,5,21}
    """
    df = df.copy()
    
    def _log_ratio(num, den):
        return np.log((num + eps) / (den + eps))
    
    new_cols = {}
    
    # VIX features
    vix_cols = ["^VIX_Open", "^VIX_High", "^VIX_Low", "^VIX_Close"]
    if all(c in df.columns for c in vix_cols):
        vix_o = df["^VIX_Open"].astype("float64")
        vix_h = df["^VIX_High"].astype("float64")
        vix_l = df["^VIX_Low"].astype("float64")
        vix_c = df["^VIX_Close"].astype("float64")
        
        new_cols["VIX_log_level"] = np.log(vix_c + eps)
        new_cols["VIX_delta_1d"] = vix_c.diff(1)
        new_cols["VIX_abs_delta_1d"] = new_cols["VIX_delta_1d"].abs()
        new_cols["VIX_log_hl"] = _log_ratio(vix_h, vix_l)
        new_cols["VIX_range_frac"] = (vix_h - vix_l) / (vix_c.abs() + eps)
        new_cols["VIX_gap_oc"] = _log_ratio(vix_c, vix_o)
        
        for lag in [1, 5, 21]:
            new_cols[f"VIX_delta_1d_lag{lag}"] = new_cols["VIX_delta_1d"].shift(lag)
            new_cols[f"VIX_abs_delta_1d_lag{lag}"] = new_cols["VIX_abs_delta_1d"].shift(lag)
    else:
        print("[WARN] Missing VIX OHLC columns, skipping VIX features")
    
    # TNX features
    tnx_cols = ["^TNX_Open", "^TNX_High", "^TNX_Low", "^TNX_Close"]
    if all(c in df.columns for c in tnx_cols):
        tnx_o = df["^TNX_Open"].astype("float64")
        tnx_h = df["^TNX_High"].astype("float64")
        tnx_l = df["^TNX_Low"].astype("float64")
        tnx_c = df["^TNX_Close"].astype("float64")
        
        new_cols["TNX_level"] = tnx_c
        new_cols["TNX_delta_1d"] = tnx_c.diff(1)
        new_cols["TNX_abs_delta_1d"] = new_cols["TNX_delta_1d"].abs()
        new_cols["TNX_delta_5d"] = tnx_c.diff(5)
        new_cols["TNX_log_hl"] = _log_ratio(tnx_h, tnx_l)
        new_cols["TNX_range"] = (tnx_h - tnx_l)
        new_cols["TNX_gap_oc"] = _log_ratio(tnx_c, tnx_o)
        
        for lag in [1, 5, 21]:
            new_cols[f"TNX_delta_1d_lag{lag}"] = new_cols["TNX_delta_1d"].shift(lag)
            new_cols[f"TNX_abs_delta_1d_lag{lag}"] = new_cols["TNX_abs_delta_1d"].shift(lag)
    else:
        print("[WARN] Missing TNX OHLC columns, skipping TNX features")
    
    # Attach
    vix_tnx_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, vix_tnx_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} VIX/TNX columns")
    return df


# ==============================================================================
# CELL 28: EVENT / SPARSE SIGNAL FEATURES
# ==============================================================================

def _prev_event_value_to_daily(
    df: pd.DataFrame,
    value_col: str,
    release_flag_col: str,
    lag_k: int = 1,
    fill_before_first: float = 0.0
) -> pd.Series:
    """
    Returns a daily series where each day carries the value from the previous (lag_k) RELEASE EVENT.
    """
    events = df.loc[df[release_flag_col] == 1, value_col].astype("float64").copy()
    shifted = events.shift(lag_k)
    
    out = pd.Series(index=df.index, dtype="float64")
    out.loc[shifted.index] = shifted.values
    out = out.ffill().fillna(fill_before_first)
    return out


def _eps_event_lags_to_daily(
    df: pd.DataFrame,
    val_col: str,
    flag_col: str,
    lag_k: int
) -> pd.Series:
    """
    Event-based lag for EPS: values only when has_eps_surprise_* == 1, shift by EVENTS.
    """
    event_series = df.loc[df[flag_col] == 1, val_col].astype("float64").copy()
    shifted = event_series.shift(lag_k)
    
    out = pd.Series(index=df.index, dtype="float64")
    out.loc[shifted.index] = shifted.values
    out = out.ffill().fillna(0.0)
    return out


def add_event_features(
    df: pd.DataFrame,
    base: str = "GOOGL",
    w_long: int = 21,
    market_vol_ticker: str = "SPY"
) -> pd.DataFrame:
    """
    Add event-based and sparse signal features.
    Matches notebook Cell 28 (BLOCK 11).
    
    Creates:
    - eps_surprise_{yahoo,calc}_lag{1,2}
    - post_earnings_day_{1-5}
    - eps_flag_yahoo_x_vol, eps_flag_calc_x_vol
    - CPI_impulse, CPI_prev_release, CPI_change_prev_release
    - FEDFUNDS_impulse, FEDFUNDS_prev_delta, FEDFUNDS_release_day_x_mkt_vol
    """
    df = df.copy()
    
    base_vol_col = f"{base}_logret_cc_std_{w_long}"
    market_vol_col = f"{market_vol_ticker}_logret_cc_std_{w_long}"
    
    new_cols = {}
    
    # EPS surprise lags by EARNINGS EVENTS
    for src in ["yahoo", "calc"]:
        val_col = f"eps_surprise_pct_{src}"
        flag_col = f"has_eps_surprise_{src}"
        
        if val_col in df.columns and flag_col in df.columns:
            new_cols[f"eps_surprise_{src}_lag1"] = _eps_event_lags_to_daily(df, val_col, flag_col, lag_k=1)
            new_cols[f"eps_surprise_{src}_lag2"] = _eps_event_lags_to_daily(df, val_col, flag_col, lag_k=2)
    
    # Post-earnings day dummies
    if "is_earnings_day" in df.columns:
        earn = df["is_earnings_day"].astype("int8")
        for k in range(1, 6):
            new_cols[f"post_earnings_day_{k}"] = earn.shift(k).fillna(0).astype("int8")
    
    # EPS flag × vol
    if base_vol_col in df.columns:
        vol_base = df[base_vol_col].astype("float64")
        
        if "has_eps_surprise_yahoo" in df.columns:
            new_cols["eps_flag_yahoo_x_vol"] = df["has_eps_surprise_yahoo"].astype("float64") * vol_base
        if "has_eps_surprise_calc" in df.columns:
            new_cols["eps_flag_calc_x_vol"] = df["has_eps_surprise_calc"].astype("float64") * vol_base
    
    # Macro: CPI (impulse + previous release)
    if "CPI_release_day" in df.columns and "CPI_pct_mom" in df.columns:
        cpi_release = df["CPI_release_day"].astype("int8")
        cpi_val = df["CPI_pct_mom"].astype("float64")
        
        # Impulse only on release day
        new_cols["CPI_impulse"] = cpi_val * cpi_release.astype("float64")
        
        # Previous release value carried forward
        new_cols["CPI_prev_release"] = _prev_event_value_to_daily(
            df, value_col="CPI_pct_mom", release_flag_col="CPI_release_day", 
            lag_k=1, fill_before_first=0.0
        )
        
        # Change vs previous release
        new_cols["CPI_change_prev_release"] = (
            (cpi_val - new_cols["CPI_prev_release"]) * cpi_release.astype("float64")
        )
    
    # Macro: FEDFUNDS (impulse + previous decision delta)
    if "FEDFUNDS_release_day" in df.columns and "FEDFUNDS_delta_mom" in df.columns:
        ff_release = df["FEDFUNDS_release_day"].astype("int8")
        ff_delta = df["FEDFUNDS_delta_mom"].astype("float64")
        
        new_cols["FEDFUNDS_impulse"] = ff_delta * ff_release.astype("float64")
        
        new_cols["FEDFUNDS_prev_delta"] = _prev_event_value_to_daily(
            df, value_col="FEDFUNDS_delta_mom", release_flag_col="FEDFUNDS_release_day",
            lag_k=1, fill_before_first=0.0
        )
        
        # Release-day × market vol
        if market_vol_col in df.columns:
            mkt_vol = df[market_vol_col].astype("float64")
            new_cols["FEDFUNDS_release_day_x_mkt_vol"] = ff_release.astype("float64") * mkt_vol
    
    # Attach
    blk11_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, blk11_df], axis=1)
    
    print(f"[INFO] Added {len(new_cols)} event/sparse-signal columns")
    return df


# ==============================================================================
# CELL 30: PERIOD FLAGS
# ==============================================================================

def add_period_flags(
    df: pd.DataFrame,
    covid_start: str = "2020-02-20",
    covid_end: str = "2020-06-01",
    crisis_2008_start: str = "2008-09-01",
    crisis_2008_end: str = "2009-06-01"
) -> pd.DataFrame:
    """
    Add period flags for COVID and 2008 crisis.
    Matches notebook Cell 30 (BLOCK 12).
    
    Creates:
    - covid_period, pre_covid, post_covid
    - crisis_2008, pre_crisis_2008, post_crisis_2008
    """
    df = df.copy()
    
    covid_start = pd.Timestamp(covid_start)
    covid_end = pd.Timestamp(covid_end)
    crisis_2008_start = pd.Timestamp(crisis_2008_start)
    crisis_2008_end = pd.Timestamp(crisis_2008_end)
    
    # COVID flags
    df["covid_period"] = ((df.index >= covid_start) & (df.index <= covid_end)).astype("int8")
    df["pre_covid"] = (df.index < covid_start).astype("int8")
    df["post_covid"] = (df.index > covid_end).astype("int8")
    
    # 2008 crisis flags
    df["crisis_2008"] = ((df.index >= crisis_2008_start) & (df.index <= crisis_2008_end)).astype("int8")
    df["pre_crisis_2008"] = (df.index < crisis_2008_start).astype("int8")
    df["post_crisis_2008"] = (df.index > crisis_2008_end).astype("int8")
    
    # Sanity checks
    bad_covid = int(((df["pre_covid"] + df["covid_period"] + df["post_covid"]) != 1).sum())
    bad_2008 = int(((df["pre_crisis_2008"] + df["crisis_2008"] + df["post_crisis_2008"]) != 1).sum())
    
    if bad_covid > 0:
        print(f"[WARN] COVID flags not mutually exclusive: {bad_covid} rows")
    if bad_2008 > 0:
        print(f"[WARN] Crisis flags not mutually exclusive: {bad_2008} rows")
    
    print(f"[INFO] Added 6 period flag columns")
    return df


# ==============================================================================
# CELL 34: DUPLICATE COLUMN REMOVAL
# ==============================================================================

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate column names (keep last).
    Matches notebook Cell 34 (BLOCK 14).
    """
    df = df.copy()
    
    dup_mask = df.columns.duplicated(keep="last")
    dup_cols = df.columns[dup_mask].tolist()
    
    if len(dup_cols) == 0:
        print("[OK] No duplicate column names found")
    else:
        before_shape = df.shape
        df = df.loc[:, ~dup_mask].copy()
        after_shape = df.shape
        print(f"[WARN] Dropped {len(dup_cols)} duplicate columns (keep_last)")
        print(f"[INFO] Shape: {before_shape} -> {after_shape}")
    
    return df


# ==============================================================================
# TARGET COLUMN
# ==============================================================================

def add_target(
    df: pd.DataFrame,
    base: str = "GOOGL",
    horizon: int = 1
) -> pd.DataFrame:
    """
    Add target column (forward return).
    
    Creates:
    - {base}_logret_t{horizon}: forward log return shifted by -horizon
    """
    df = df.copy()
    
    logret_col = f"{base}_logret_cc"
    if logret_col not in df.columns:
        raise ValueError(f"Missing {logret_col} for target creation")
    
    target_col = f"{base}_logret_t{horizon}"
    df[target_col] = df[logret_col].shift(-horizon)
    
    print(f"[INFO] Added target column {target_col} (horizon={horizon})")
    return df


# ==============================================================================
# BUILD ALL FEATURES
# ==============================================================================

def build_all_features(
    df: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Build all features in sequence.
    Matches notebook feature engineering pipeline (Cells 18-34).
    
    Args:
        df: Raw DataFrame with price/earnings/macro data
        config: Configuration dict with feature parameters
    
    Returns:
        DataFrame with all features added
    """
    df = df.copy().sort_index()
    
    # Get config parameters
    feat_cfg = config.get("features", {})
    eps = float(feat_cfg.get("eps", EPS))
    w_short = int(feat_cfg.get("rolling_w_short", 5))
    w_long = int(feat_cfg.get("rolling_w_long", 21))
    base = str(feat_cfg.get("regime_base", "GOOGL"))
    peers = list(feat_cfg.get("cross_asset_peers", ["SPY", "QQQ", "MSFT", "NVDA"]))
    windows = list(feat_cfg.get("cross_asset_windows", [5, 21]))
    exclude_raw_ohlc = list(feat_cfg.get("exclude_raw_ohlc", ["^VIX", "^TNX"]))
    do_volume_rolling = bool(feat_cfg.get("do_volume_rolling", True))
    market_vol_ticker = str(feat_cfg.get("market_vol_ticker", "SPY"))
    target_horizon = int(feat_cfg.get("target_horizon", 1))
    
    # Period flags config
    covid_start = str(feat_cfg.get("covid_start", "2020-02-20"))
    covid_end = str(feat_cfg.get("covid_end", "2020-06-01"))
    crisis_2008_start = str(feat_cfg.get("crisis_2008_start", "2008-09-01"))
    crisis_2008_end = str(feat_cfg.get("crisis_2008_end", "2009-06-01"))
    
    print(f"[INFO] Building features (base={base}, w_short={w_short}, w_long={w_long})")
    
    # Drop raw Volume columns for specified tickers (they have unreliable/no volume data)
    drop_volume_tickers = list(feat_cfg.get("drop_volume_tickers", []))
    if drop_volume_tickers:
        drop_volume_cols = [f"{t}_Volume" for t in drop_volume_tickers]
        existing_drop_cols = [c for c in drop_volume_cols if c in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
            print(f"[INFO] Dropped {len(existing_drop_cols)} unreliable Volume columns: {existing_drop_cols}")
    
    # Cell 18: Raw transformations
    df = add_raw_transformations(df, exclude_raw_ohlc=exclude_raw_ohlc, eps=eps)
    
    # Cell 20: Rolling statistics
    df = add_rolling_statistics(df, w_short=w_short, w_long=w_long, 
                                do_volume_rolling=do_volume_rolling, eps=eps)
    
    # Cell 22: Cross-asset features
    df = add_cross_asset_features(df, base=base, peers=peers, windows=windows, eps=eps)
    
    # Cell 24: Regime features
    df = add_regime_features(df, base=base, w_short=w_short, w_long=w_long, 
                             peers=peers, eps=eps)
    
    # Cell 26: VIX/TNX features
    df = add_vix_tnx_features(df, eps=eps)
    
    # Cell 28: Event features
    df = add_event_features(df, base=base, w_long=w_long, 
                            market_vol_ticker=market_vol_ticker)
    
    # Cell 30: Period flags
    df = add_period_flags(df, covid_start=covid_start, covid_end=covid_end,
                          crisis_2008_start=crisis_2008_start, crisis_2008_end=crisis_2008_end)
    
    # Cell 32: Target
    df = add_target(df, base=base, horizon=target_horizon)
    
    # Cell 34: Remove duplicate columns
    df = remove_duplicate_columns(df)
    
    print(f"[INFO] Feature engineering complete. Shape: {df.shape}")
    return df
