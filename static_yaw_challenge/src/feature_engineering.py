"""
Feature Engineering Module for Static Yaw Challenge

This module contains functions for creating features from SCADA telemetry data
to identify static yaw offsets in wind turbine operations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Turbine specifications
TURBINE_SPECS = {
    'rated_power_kw': 6.2,
    'rotor_diameter_m': 12.9,
    'hub_height_m': 18.0,
    'gear_ratio': 12.0,
    'air_density': 1.225  # kg/m³ at sea level
}


def calculate_rotor_area(diameter_m: float = TURBINE_SPECS['rotor_diameter_m']) -> float:
    """
    Calculate rotor swept area.

    Parameters
    ----------
    diameter_m : float
        Rotor diameter in meters

    Returns
    -------
    float
        Swept area in m²
    """
    return np.pi * (diameter_m / 2) ** 2


def calculate_power_coefficient(power_kw: np.ndarray,
                                wind_speed_ms: np.ndarray,
                                air_density: float = TURBINE_SPECS['air_density'],
                                rotor_area: float = None) -> np.ndarray:
    """
    Calculate power coefficient (Cp) - efficiency of power extraction.

    Cp = Actual Power / Available Wind Power
    Available Wind Power = 0.5 * ρ * A * v³

    Parameters
    ----------
    power_kw : np.ndarray
        Actual power output in kW
    wind_speed_ms : np.ndarray
        Wind speed in m/s
    air_density : float
        Air density in kg/m³
    rotor_area : float
        Rotor swept area in m² (calculated if not provided)

    Returns
    -------
    np.ndarray
        Power coefficient (dimensionless, theoretical max ~0.59)
    """
    if rotor_area is None:
        rotor_area = calculate_rotor_area()

    # Available wind power in kW
    wind_power_kw = 0.5 * air_density * rotor_area * (wind_speed_ms ** 3) / 1000

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cp = power_kw / wind_power_kw
        cp = np.where(wind_power_kw > 0.01, cp, 0)  # Set to 0 for very low wind
        cp = np.clip(cp, 0, 1)  # Cp cannot exceed 1

    return cp


def calculate_tip_speed_ratio(rotor_speed_rpm: np.ndarray,
                              wind_speed_ms: np.ndarray,
                              rotor_diameter_m: float = TURBINE_SPECS['rotor_diameter_m']) -> np.ndarray:
    """
    Calculate tip speed ratio (TSR) - ratio of blade tip speed to wind speed.

    TSR = (ω * R) / v
    where ω is angular velocity, R is rotor radius, v is wind speed

    Parameters
    ----------
    rotor_speed_rpm : np.ndarray
        Rotor speed in RPM
    wind_speed_ms : np.ndarray
        Wind speed in m/s
    rotor_diameter_m : float
        Rotor diameter in meters

    Returns
    -------
    np.ndarray
        Tip speed ratio (dimensionless)
    """
    rotor_radius_m = rotor_diameter_m / 2
    # Convert RPM to rad/s: RPM * 2π / 60
    angular_velocity = rotor_speed_rpm * 2 * np.pi / 60
    tip_speed = angular_velocity * rotor_radius_m

    with np.errstate(divide='ignore', invalid='ignore'):
        tsr = tip_speed / wind_speed_ms
        tsr = np.where(wind_speed_ms > 0.5, tsr, 0)  # Set to 0 for very low wind

    return tsr


def calculate_power_loss_features(df: pd.DataFrame,
                                  baseline_yaw: float = 0.0,
                                  wind_bins: np.ndarray = None) -> pd.DataFrame:
    """
    Calculate power loss features relative to baseline yaw alignment.

    This creates a power curve for baseline yaw (typically 0°) and compares
    actual power to expected power for each wind speed bin.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with 'wind_speed', 'power_output', 'yaw_offset'
    baseline_yaw : float
        Baseline yaw offset (typically 0°)
    wind_bins : np.ndarray
        Wind speed bin edges (default: 0.5 m/s bins from 0-25 m/s)

    Returns
    -------
    pd.DataFrame
        Original dataframe with added power loss features
    """
    if wind_bins is None:
        wind_bins = np.arange(0, 25, 0.5)

    df = df.copy()

    # Create baseline power curve (from 0° yaw data)
    baseline_data = df[df['yaw_offset'] == baseline_yaw].copy()
    baseline_data['wind_bin'] = pd.cut(baseline_data['wind_speed'], bins=wind_bins)
    baseline_power_curve = baseline_data.groupby('wind_bin')['power_output'].mean()

    # Map baseline power to all rows based on wind speed
    df['wind_bin'] = pd.cut(df['wind_speed'], bins=wind_bins)
    df['expected_power'] = df['wind_bin'].map(baseline_power_curve)

    # Calculate power loss metrics
    df['power_loss_kw'] = df['expected_power'] - df['power_output']
    df['power_loss_pct'] = (df['power_loss_kw'] / df['expected_power'] * 100).fillna(0)
    df['power_efficiency'] = (df['power_output'] / df['expected_power']).fillna(1).clip(0, 1.5)

    df = df.drop(columns=['wind_bin'])

    return df


def calculate_circular_statistics(angles_deg: np.ndarray) -> Dict[str, float]:
    """
    Calculate circular statistics for wind direction data.

    Parameters
    ----------
    angles_deg : np.ndarray
        Angles in degrees

    Returns
    -------
    dict
        Dictionary with circular mean, std, and dispersion
    """
    # Convert to radians
    angles_rad = np.deg2rad(angles_deg)

    # Circular mean
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    circular_mean_rad = np.arctan2(sin_mean, cos_mean)
    circular_mean_deg = np.rad2deg(circular_mean_rad)

    # Circular standard deviation
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_std_rad = np.sqrt(-2 * np.log(R))
    circular_std_deg = np.rad2deg(circular_std_rad)

    # Circular dispersion (0 = no dispersion, 1 = uniform)
    circular_dispersion = 1 - R

    return {
        'circular_mean': circular_mean_deg,
        'circular_std': circular_std_deg,
        'circular_dispersion': circular_dispersion,
        'resultant_length': R
    }


def create_segment_features(df: pd.DataFrame,
                           segment_col: str = 'segment_id',
                           exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Create segment-level aggregated features from row-level data.

    This is the primary feature engineering function for segment-level modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level data with segment identifiers
    segment_col : str
        Column name for segment identifier
    exclude_cols : list
        Columns to exclude from aggregation

    Returns
    -------
    pd.DataFrame
        Segment-level features with one row per segment
    """
    if exclude_cols is None:
        exclude_cols = ['row_id', 'datetime', 'segment_time']

    # Columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols and c != segment_col]

    # Basic statistical aggregations
    agg_dict = {}
    for col in numeric_cols:
        if col != 'turbine_status':  # Handle status separately
            agg_dict[col] = ['mean', 'std', 'min', 'max', 'median']

    # Aggregate numeric features
    segment_features = df.groupby(segment_col)[numeric_cols].agg(agg_dict)
    segment_features.columns = ['_'.join(col).strip() for col in segment_features.columns.values]
    segment_features = segment_features.reset_index()

    # Turbine status features (categorical)
    status_features = df.groupby(segment_col)['turbine_status'].agg([
        ('status_mode', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]),
        ('status_nunique', 'nunique'),
        ('pct_production', lambda x: (x == 10).sum() / len(x) * 100),
        ('pct_standby', lambda x: (x == 9).sum() / len(x) * 100),
        ('pct_alarm', lambda x: (x == 13).sum() / len(x) * 100)
    ]).reset_index()

    # Merge status features
    segment_features = segment_features.merge(status_features, on=segment_col)

    # Wind direction circular statistics (if available)
    if 'relative_wind_direction' in df.columns:
        wind_dir_stats = df.groupby(segment_col)['relative_wind_direction'].apply(
            lambda x: pd.Series(calculate_circular_statistics(x.values))
        ).reset_index()
        wind_dir_stats.columns = [segment_col, 'wind_dir_circular_mean',
                                  'wind_dir_circular_std', 'wind_dir_circular_dispersion',
                                  'wind_dir_resultant_length']
        segment_features = segment_features.merge(wind_dir_stats, on=segment_col)

    # Power coefficient features (if power and wind speed available)
    if 'power_output' in df.columns and 'wind_speed' in df.columns:
        df_copy = df.copy()
        df_copy['power_coefficient'] = calculate_power_coefficient(
            df_copy['power_output'].values,
            df_copy['wind_speed'].values
        )
        cp_stats = df_copy.groupby(segment_col)['power_coefficient'].agg([
            ('cp_mean', 'mean'),
            ('cp_std', 'std'),
            ('cp_max', 'max')
        ]).reset_index()
        segment_features = segment_features.merge(cp_stats, on=segment_col)

    # Tip speed ratio features
    if 'rotor_speed' in df.columns and 'wind_speed' in df.columns:
        df_copy = df.copy()
        df_copy['tip_speed_ratio'] = calculate_tip_speed_ratio(
            df_copy['rotor_speed'].values,
            df_copy['wind_speed'].values
        )
        tsr_stats = df_copy.groupby(segment_col)['tip_speed_ratio'].agg([
            ('tsr_mean', 'mean'),
            ('tsr_std', 'std')
        ]).reset_index()
        segment_features = segment_features.merge(tsr_stats, on=segment_col)

    return segment_features


def create_rolling_features(df: pd.DataFrame,
                           windows: List[int] = None,
                           features: List[str] = None) -> pd.DataFrame:
    """
    Create rolling window features for time series analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data (must be sorted by time)
    windows : list
        Window sizes in number of rows (default: [10, 30, 60, 300] for 1Hz data)
    features : list
        Columns to create rolling features for

    Returns
    -------
    pd.DataFrame
        Original dataframe with added rolling features
    """
    if windows is None:
        windows = [10, 30, 60, 300]  # 10s, 30s, 1min, 5min at 1Hz

    if features is None:
        features = ['wind_speed', 'power_output', 'rotor_speed']

    df = df.copy()

    for feature in features:
        if feature not in df.columns:
            continue

        for window in windows:
            # Rolling mean
            df[f'{feature}_roll_mean_{window}'] = df[feature].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling std
            df[f'{feature}_roll_std_{window}'] = df[feature].rolling(
                window=window, min_periods=1
            ).std()

            # Rate of change
            if window >= 10:
                df[f'{feature}_roc_{window}'] = df[feature].diff(window) / window

    return df


def create_temporal_features(df: pd.DataFrame,
                            datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Create temporal features from datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Data with datetime column
    datetime_col : str
        Name of datetime column

    Returns
    -------
    pd.DataFrame
        Original dataframe with added temporal features
    """
    df = df.copy()

    if datetime_col not in df.columns:
        return df

    # Ensure datetime type
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Extract temporal components
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['day_of_year'] = df[datetime_col].dt.dayofyear
    df['month'] = df[datetime_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical encoding for day of year
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df


def filter_production_mode(df: pd.DataFrame,
                          status_col: str = 'turbine_status',
                          production_status: int = 10) -> pd.DataFrame:
    """
    Filter data to production mode only.

    Parameters
    ----------
    df : pd.DataFrame
        Data with turbine status
    status_col : str
        Column name for turbine status
    production_status : int
        Status code for production mode

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return df[df[status_col] == production_status].copy()


def create_power_curve_deviation(df: pd.DataFrame,
                                 reference_curve: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate deviation from reference power curve.

    Parameters
    ----------
    df : pd.DataFrame
        Data with wind_speed and power_output
    reference_curve : pd.DataFrame
        Reference power curve (wind_speed, expected_power)
        If None, uses ideal theoretical curve

    Returns
    -------
    pd.DataFrame
        Original dataframe with power curve deviation features
    """
    df = df.copy()

    # Calculate ideal power (simplified model)
    # P = 0.5 * Cp * ρ * A * v³ with assumed Cp=0.45 for optimal alignment
    if reference_curve is None:
        rotor_area = calculate_rotor_area()
        ideal_cp = 0.45
        air_density = TURBINE_SPECS['air_density']

        df['ideal_power_kw'] = (0.5 * ideal_cp * air_density * rotor_area *
                               (df['wind_speed'] ** 3) / 1000)
        df['ideal_power_kw'] = df['ideal_power_kw'].clip(0, TURBINE_SPECS['rated_power_kw'])
    else:
        # Use provided reference curve
        df = df.merge(reference_curve, on='wind_speed', how='left')
        df['ideal_power_kw'] = df['expected_power']

    # Deviation metrics
    df['power_deviation_kw'] = df['power_output'] - df['ideal_power_kw']
    df['power_deviation_pct'] = (df['power_deviation_kw'] /
                                 (df['ideal_power_kw'] + 1e-6) * 100).clip(-100, 100)

    return df


def add_lag_features(df: pd.DataFrame,
                    features: List[str],
                    lags: List[int] = None,
                    group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Add lag features for time series modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    features : list
        Features to create lags for
    lags : list
        Lag periods (default: [1, 5, 10, 60])
    group_col : str
        Column to group by (e.g., segment_id) to avoid cross-segment lags

    Returns
    -------
    pd.DataFrame
        Original dataframe with lag features
    """
    if lags is None:
        lags = [1, 5, 10, 60]

    df = df.copy()

    for feature in features:
        if feature not in df.columns:
            continue

        for lag in lags:
            if group_col:
                df[f'{feature}_lag_{lag}'] = df.groupby(group_col)[feature].shift(lag)
            else:
                df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    return df


def get_feature_importance_names(segment_features: pd.DataFrame,
                                exclude_cols: List[str] = None) -> List[str]:
    """
    Get list of feature names for modeling (excluding IDs and targets).

    Parameters
    ----------
    segment_features : pd.DataFrame
        Segment-level features
    exclude_cols : list
        Additional columns to exclude

    Returns
    -------
    list
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = []

    exclude_default = ['segment_id', 'row_id', 'datetime', 'yaw_offset']
    exclude_cols = exclude_default + exclude_cols

    feature_cols = [c for c in segment_features.columns if c not in exclude_cols]

    return feature_cols


# Example usage documentation
if __name__ == "__main__":
    print("Feature Engineering Module for Static Yaw Challenge")
    print("=" * 60)
    print("\nKey Functions:")
    print("  - create_segment_features(): Primary function for segment-level features")
    print("  - calculate_power_coefficient(): Calculate Cp from power and wind speed")
    print("  - calculate_tip_speed_ratio(): Calculate TSR")
    print("  - calculate_power_loss_features(): Compare to baseline power curve")
    print("  - create_rolling_features(): Rolling window statistics")
    print("  - create_temporal_features(): Extract datetime features")
    print("  - filter_production_mode(): Filter to production data only")
    print("\nTurbine Specifications:")
    for key, value in TURBINE_SPECS.items():
        print(f"  - {key}: {value}")
